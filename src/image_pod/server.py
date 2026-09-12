from __future__ import annotations

import hmac
import json
import os
import shutil
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from src.image_pod.media import cleanup_finalized_outputs, finalized_output_path
from src.image_pod.registry import get_adapter, supported_task_families


HOST = os.environ.get("IMAGE_POD_HOST", "0.0.0.0")
PORT = int(os.environ.get("IMAGE_POD_PORT", "8000"))
POD_TOKEN = os.environ.get("SCENEBUILDER_POD_TOKEN", "").strip()
WORKER_ID = os.environ.get("SCENEBUILDER_WORKER_ID", "").strip()
CONTROL_URL = os.environ.get("SCENEBUILDER_CONTROL_URL", "").strip()
DEFAULT_IDLE_TIMEOUT = max(0, int(os.environ.get("IMAGE_POD_IDLE_TIMEOUT_SECONDS", "60")))
IDLE_RENEW_GRACE_SECONDS = max(5, int(os.environ.get("IMAGE_POD_IDLE_RENEW_GRACE_SECONDS", "30")))
REQUEST_MAX_BYTES = max(1024, int(os.environ.get("IMAGE_POD_MAX_REQUEST_BYTES", str(1024 * 1024))))
HEARTBEAT_SECONDS = max(5, int(os.environ.get("IMAGE_POD_HEARTBEAT_SECONDS", "15")))
JOB_HISTORY_MAX = max(1, int(os.environ.get("IMAGE_POD_JOB_HISTORY_MAX", "100")))
JOB_HISTORY_TTL_SECONDS = max(60, int(os.environ.get("IMAGE_POD_JOB_HISTORY_TTL_SECONDS", "3600")))
TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled"}


class RequestTooLargeError(ValueError):
    pass


@dataclass
class JobRecord:
    job_id: str
    task_family: str
    status: str = "queued"
    phase: str = "queued"
    progress_percent: int | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    cancel_requested: bool = False

    def public(self) -> dict[str, Any]:
        return {
            "jobId": self.job_id,
            "taskFamily": self.task_family,
            "status": self.status,
            "phase": self.phase,
            "progressPercent": self.progress_percent,
            "createdAt": self.created_at,
            "startedAt": self.started_at,
            "completedAt": self.completed_at,
            "result": self.result,
            "error": self.error,
            "cancelRequested": self.cancel_requested,
        }


class ImagePodState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.jobs: dict[str, JobRecord] = {}
        self.current_job_id: str | None = None
        self.loaded_family: str | None = None
        self.worker_status = "starting"
        self.draining = False
        self.started_at = time.time()
        self.idle_since: float | None = None
        self.terminate_after: float | None = None
        self.idle_timeout_seconds = DEFAULT_IDLE_TIMEOUT

    def mark_idle(self) -> None:
        now = time.time()
        with self.lock:
            if self.draining or self.worker_status == "unhealthy":
                return
            self.current_job_id = None
            self.worker_status = "idle"
            self.idle_since = now
            self.terminate_after = now + self.idle_timeout_seconds
        emit_event(
            "worker_idle",
            idleSince=now,
            idleTimeoutSeconds=self.idle_timeout_seconds,
            terminateAfter=now + self.idle_timeout_seconds,
            loadedFamily=self.loaded_family,
        )

    def renew_idle(self) -> dict[str, Any] | None:
        now = time.time()
        with self.lock:
            if self.current_job_id is not None or self.worker_status == "unhealthy":
                return None
            active_idle = self.worker_status == "idle" and not self.draining
            recent_idle_expiry = (
                self.worker_status == "draining"
                and self.draining
                and self.terminate_after is not None
                and now <= self.terminate_after + IDLE_RENEW_GRACE_SECONDS
            )
            if not active_idle and not recent_idle_expiry:
                return None
            self.draining = False
            self.worker_status = "idle"
            self.idle_since = now
            self.terminate_after = now + self.idle_timeout_seconds
            return {
                "ok": True,
                "status": "idle",
                "idleSince": self.idle_since,
                "idleTimeoutSeconds": self.idle_timeout_seconds,
                "terminateAfter": self.terminate_after,
                "loadedFamily": self.loaded_family,
            }

    def mark_unhealthy(self, reason: str) -> None:
        with self.lock:
            self.current_job_id = None
            self.worker_status = "unhealthy"
            self.draining = True
            self.idle_since = None
            self.terminate_after = None
        emit_event("worker_unhealthy", reason=reason, loadedFamily=self.loaded_family)

    def prune_jobs(self) -> None:
        cutoff = time.time() - JOB_HISTORY_TTL_SECONDS
        with self.lock:
            terminal = [
                record
                for record in self.jobs.values()
                if record.status in TERMINAL_JOB_STATUSES and record.completed_at is not None
            ]
            for record in terminal:
                if record.completed_at is not None and record.completed_at < cutoff:
                    cleanup_finalized_outputs(record.job_id)
                    self.jobs.pop(record.job_id, None)
            terminal = sorted(
                (
                    record
                    for record in self.jobs.values()
                    if record.status in TERMINAL_JOB_STATUSES and record.completed_at is not None
                ),
                key=lambda record: record.completed_at or 0,
                reverse=True,
            )
            for record in terminal[JOB_HISTORY_MAX:]:
                cleanup_finalized_outputs(record.job_id)
                self.jobs.pop(record.job_id, None)


STATE = ImagePodState()


def _required_runtime_config_errors() -> list[str]:
    required = {
        "SCENEBUILDER_POD_TOKEN": POD_TOKEN,
        "SCENEBUILDER_WORKER_ID": WORKER_ID,
        "SCENEBUILDER_CONTROL_URL": CONTROL_URL,
    }
    return [name for name, value in required.items() if not value]


def _json_request(url: str, payload: dict[str, Any], timeout: int = 10) -> None:
    if not url:
        return
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": "SceneBuilder-Image-Pod/1.0"}
    if POD_TOKEN:
        headers["Authorization"] = f"Bearer {POD_TOKEN}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read(1024)
    except Exception as exc:
        print(f"[Image Pod] control callback failed: {type(exc).__name__}: {exc}", flush=True)


def emit_event(event_type: str, **fields: Any) -> None:
    if not CONTROL_URL:
        return
    payload: dict[str, Any] = {
        "event": event_type,
        "eventId": str(uuid.uuid4()),
        "nonce": uuid.uuid4().hex,
        "workerId": WORKER_ID,
        "timestamp": time.time(),
        "timestampMs": int(time.time() * 1000),
    }
    payload.update(fields)
    threading.Thread(target=_json_request, args=(CONTROL_URL, payload), daemon=True).start()


def _progress_callback(record: JobRecord):
    def callback(phase: str, percent: int | None = None) -> None:
        with STATE.lock:
            record.phase = str(phase)
            record.progress_percent = None if percent is None else max(0, min(100, int(percent)))
        emit_event(
            "job_progress",
            jobId=record.job_id,
            taskFamily=record.task_family,
            phase=record.phase,
            progressPercent=record.progress_percent,
        )

    return callback


def _cancel_requested(record: JobRecord) -> bool:
    with STATE.lock:
        return bool(record.cancel_requested)


def _is_fatal_runtime_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    fatal_markers = (
        "cuda out of memory",
        "outofmemoryerror",
        "cuda unavailable",
        "cuda error",
        "cublas",
        "cudnn",
        "illegal memory access",
        "device-side assert",
        "comfyui exited during startup",
        "comfyui did not become ready",
        "failed to load diffusion",
        "failed to load clip",
        "failed to load vae",
        "model load",
        "text encoder load",
        "vae load",
    )
    return any(marker in text for marker in fatal_markers)


def process_job(payload: dict[str, Any], record: JobRecord) -> None:
    adapter = get_adapter(record.task_family)
    fatal = False
    with STATE.lock:
        record.status = "processing"
        record.phase = "preparing_model"
        record.started_at = time.time()
        STATE.worker_status = "busy"
        STATE.current_job_id = record.job_id
        STATE.idle_since = None
        STATE.terminate_after = None

    emit_event("job_started", jobId=record.job_id, taskFamily=record.task_family)
    try:
        result = adapter.run_job(
            payload,
            progress=_progress_callback(record),
            cancel_requested=lambda: _cancel_requested(record),
        )
        with STATE.lock:
            if record.cancel_requested:
                record.status = "cancelled"
                record.phase = "cancelled"
            else:
                record.status = "completed"
                record.phase = "completed"
                record.progress_percent = 100
                record.result = result
                STATE.loaded_family = record.task_family
        if record.status == "cancelled":
            emit_event("job_cancelled", jobId=record.job_id, taskFamily=record.task_family)
        else:
            emit_event("job_completed", jobId=record.job_id, taskFamily=record.task_family, result=result)
    except Exception as exc:
        cancelled = _cancel_requested(record)
        fatal = not cancelled and _is_fatal_runtime_error(exc)
        with STATE.lock:
            record.status = "cancelled" if cancelled else "failed"
            record.phase = record.status
            record.error = None if cancelled else {
                "code": "IMAGE_RUNTIME_FATAL" if fatal else "IMAGE_RUNTIME_ERROR",
                "message": str(exc),
                "type": type(exc).__name__,
            }
        if cancelled:
            emit_event("job_cancelled", jobId=record.job_id, taskFamily=record.task_family)
        else:
            emit_event(
                "job_failed",
                jobId=record.job_id,
                taskFamily=record.task_family,
                error=record.error,
                workerFatal=fatal,
            )
            print(f"[Image Pod] job {record.job_id} failed: {type(exc).__name__}: {exc}", flush=True)
    finally:
        with STATE.lock:
            record.completed_at = time.time()
        STATE.prune_jobs()
        if fatal:
            STATE.mark_unhealthy(record.error["message"] if record.error else "fatal runtime error")
        else:
            STATE.mark_idle()


def idle_watchdog() -> None:
    while True:
        time.sleep(1)
        expired = False
        deadline = None
        with STATE.lock:
            deadline = STATE.terminate_after
            if (
                not STATE.draining
                and STATE.worker_status == "idle"
                and deadline is not None
                and time.time() >= deadline
            ):
                STATE.draining = True
                STATE.worker_status = "draining"
                expired = True
        if expired:
            emit_event("idle_expired", terminateAfter=deadline, loadedFamily=STATE.loaded_family)


def heartbeat_loop() -> None:
    while True:
        time.sleep(HEARTBEAT_SECONDS)
        with STATE.lock:
            fields = {
                "status": STATE.worker_status,
                "currentJobId": STATE.current_job_id,
                "loadedFamily": STATE.loaded_family,
                "draining": STATE.draining,
            }
        emit_event("worker_heartbeat", **fields)


def gpu_diagnostics() -> dict[str, Any]:
    try:
        import torch

        data: dict[str, Any] = {
            "torchVersion": torch.__version__,
            "torchCudaVersion": torch.version.cuda,
            "cudaAvailable": bool(torch.cuda.is_available()),
        }
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            props = torch.cuda.get_device_properties(0)
            data.update(
                {
                    "deviceName": props.name,
                    "vramBytes": int(props.total_memory),
                    "capability": list(torch.cuda.get_device_capability(0)),
                }
            )
        return data
    except Exception as exc:
        return {"cudaAvailable": False, "error": str(exc), "errorType": type(exc).__name__}


def runtime_readiness() -> dict[str, Any]:
    config_errors = _required_runtime_config_errors()
    gpu = gpu_diagnostics()
    adapter_ready = False
    adapter_details: dict[str, Any] = {}
    try:
        adapter = get_adapter("krea2_image")
        readiness = getattr(adapter, "readiness", None)
        if callable(readiness):
            adapter_details = dict(readiness())
            adapter_ready = bool(adapter_details.get("ready"))
        else:
            adapter_ready = bool(adapter.is_ready())
    except Exception as exc:
        adapter_details = {"ready": False, "error": str(exc), "errorType": type(exc).__name__}

    with STATE.lock:
        state_ready = STATE.worker_status in {"idle", "busy"} and not STATE.draining
        status = STATE.worker_status
    ready = not config_errors and bool(gpu.get("cudaAvailable")) and adapter_ready and state_ready
    return {
        "ready": ready,
        "status": status,
        "taskFamilies": supported_task_families(),
        "configErrors": config_errors,
        "gpu": gpu,
        "adapter": adapter_details,
    }


def bootstrap() -> None:
    try:
        config_errors = _required_runtime_config_errors()
        if config_errors:
            raise RuntimeError(f"missing required runtime configuration: {', '.join(config_errors)}")
        gpu = gpu_diagnostics()
        if not gpu.get("cudaAvailable"):
            raise RuntimeError(f"CUDA unavailable: {gpu}")
        adapter = get_adapter("krea2_image")
        adapter.ensure_ready()
        readiness = getattr(adapter, "readiness", None)
        if callable(readiness):
            details = dict(readiness())
            if not details.get("ready"):
                raise RuntimeError(f"Krea2 runtime assets not ready: {details}")
        with STATE.lock:
            STATE.worker_status = "idle"
        STATE.mark_idle()
        emit_event("worker_ready", taskFamilies=supported_task_families(), readyAt=time.time())
    except Exception as exc:
        STATE.mark_unhealthy(str(exc))
        print(f"[Image Pod] bootstrap failed: {type(exc).__name__}: {exc}", flush=True)


class Handler(BaseHTTPRequestHandler):
    server_version = "SceneBuilderImagePod/1.0"

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, status: int, output: Any, content_type: str) -> None:
        size = output.stat().st_size
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with output.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        if length > REQUEST_MAX_BYTES:
            raise RequestTooLargeError(f"request body exceeds {REQUEST_MAX_BYTES} bytes")
        raw = self.rfile.read(length)
        if len(raw) > REQUEST_MAX_BYTES:
            raise RequestTooLargeError(f"request body exceeds {REQUEST_MAX_BYTES} bytes")
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _authorized(self) -> bool:
        if not POD_TOKEN:
            return False
        expected = f"Bearer {POD_TOKEN}"
        supplied = self.headers.get("Authorization", "")
        return hmac.compare_digest(supplied, expected)

    def _require_auth(self) -> bool:
        if self._authorized():
            return True
        self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
        return False

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send_json(HTTPStatus.OK, {"ok": True, "service": "scene-builder-image-pod"})
            return
        if path == "/ready":
            payload = runtime_readiness()
            self._send_json(HTTPStatus.OK if payload["ready"] else HTTPStatus.SERVICE_UNAVAILABLE, payload)
            return
        if not self._require_auth():
            return
        if path == "/diagnostics":
            with STATE.lock:
                payload = {
                    "workerId": WORKER_ID,
                    "status": STATE.worker_status,
                    "currentJobId": STATE.current_job_id,
                    "loadedFamily": STATE.loaded_family,
                    "draining": STATE.draining,
                    "idleSince": STATE.idle_since,
                    "terminateAfter": STATE.terminate_after,
                    "taskFamilies": supported_task_families(),
                    "uptimeSeconds": time.time() - STATE.started_at,
                    "jobHistoryCount": len(STATE.jobs),
                }
            self._send_json(HTTPStatus.OK, payload)
            return
        if path == "/diagnostics/gpu":
            self._send_json(HTTPStatus.OK, gpu_diagnostics())
            return
        if path.startswith("/jobs/"):
            STATE.prune_jobs()
            parts = path[len("/jobs/") :].strip("/").split("/")
            job_id = parts[0]
            with STATE.lock:
                record = STATE.jobs.get(job_id)
                payload = record.public() if record else None
            if len(parts) == 3 and parts[1] == "output" and record and record.status == "completed":
                file_name = "image.png" if parts[2] == "image" else "thumbnail.jpg" if parts[2] == "thumbnail" else ""
                output = finalized_output_path(job_id, file_name)
                if output is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "output_not_found"})
                else:
                    self._send_file(
                        HTTPStatus.OK,
                        output,
                        "image/png" if file_name == "image.png" else "image/jpeg",
                    )
                return
            if payload is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "job_not_found"})
            else:
                self._send_json(HTTPStatus.OK, payload)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._require_auth():
            return
        path = self.path.split("?", 1)[0]
        if path == "/idle/renew":
            payload = STATE.renew_idle()
            if payload is None:
                self._send_json(
                    HTTPStatus.CONFLICT,
                    {
                        "error": "worker_not_idle",
                        "status": STATE.worker_status,
                        "currentJobId": STATE.current_job_id,
                    },
                )
            else:
                self._send_json(HTTPStatus.OK, payload)
            return

        if path == "/jobs":
            try:
                payload = self._read_json()
                job_id = str(payload.get("jobId") or payload.get("job_id") or "").strip()
                family = str(payload.get("taskFamily") or payload.get("task_family") or "").strip()
                if not job_id:
                    raise ValueError("jobId is required")
                if family not in supported_task_families():
                    raise ValueError(f"unsupported taskFamily: {family or '<missing>'}")

                STATE.prune_jobs()
                with STATE.lock:
                    if STATE.draining or STATE.worker_status in {"unhealthy", "draining"}:
                        self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "worker_unavailable"})
                        return
                    existing = STATE.jobs.get(job_id)
                    if existing:
                        self._send_json(HTTPStatus.OK, existing.public())
                        return
                    if STATE.worker_status != "idle" or STATE.current_job_id is not None:
                        self._send_json(
                            HTTPStatus.CONFLICT,
                            {"error": "worker_busy", "currentJobId": STATE.current_job_id},
                        )
                        return
                    record = JobRecord(job_id=job_id, task_family=family)
                    STATE.jobs[job_id] = record
                    STATE.worker_status = "busy"
                    STATE.current_job_id = job_id
                    STATE.idle_since = None
                    STATE.terminate_after = None

                thread = threading.Thread(target=process_job, args=(payload, record), daemon=True)
                thread.start()
                self._send_json(HTTPStatus.ACCEPTED, record.public())
            except RequestTooLargeError as exc:
                self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request_too_large", "message": str(exc)})
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request", "message": str(exc)})
            return

        if path.startswith("/jobs/") and path.endswith("/cancel"):
            job_id = path[len("/jobs/") : -len("/cancel")].strip("/")
            with STATE.lock:
                record = STATE.jobs.get(job_id)
                if not record:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "job_not_found"})
                    return
                record.cancel_requested = True
                running = STATE.current_job_id == job_id and record.status in {"queued", "processing"}
            if running:
                try:
                    get_adapter(record.task_family).cancel()
                except Exception:
                    pass
            self._send_json(HTTPStatus.OK, record.public())
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[Image Pod HTTP] {self.address_string()} - {fmt % args}", flush=True)


def main() -> None:
    threading.Thread(target=bootstrap, daemon=True).start()
    threading.Thread(target=idle_watchdog, daemon=True).start()
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[Image Pod] listening on {HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

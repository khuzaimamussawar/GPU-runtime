from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from src.image_pod.registry import get_adapter, supported_task_families


HOST = os.environ.get("IMAGE_POD_HOST", "0.0.0.0")
PORT = int(os.environ.get("IMAGE_POD_PORT", "8000"))
POD_TOKEN = os.environ.get("SCENEBUILDER_POD_TOKEN", "").strip()
WORKER_ID = os.environ.get("SCENEBUILDER_WORKER_ID", "").strip()
CONTROL_URL = os.environ.get("SCENEBUILDER_CONTROL_URL", "").strip()
DEFAULT_IDLE_TIMEOUT = max(0, int(os.environ.get("IMAGE_POD_IDLE_TIMEOUT_SECONDS", "60")))


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
            if self.draining:
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


STATE = ImagePodState()


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
        "workerId": WORKER_ID,
        "timestamp": time.time(),
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


def process_job(payload: dict[str, Any], record: JobRecord) -> None:
    adapter = get_adapter(record.task_family)
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
        with STATE.lock:
            record.status = "cancelled" if cancelled else "failed"
            record.phase = record.status
            record.error = None if cancelled else {
                "code": "IMAGE_RUNTIME_ERROR",
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
            )
            print(f"[Image Pod] job {record.job_id} failed: {type(exc).__name__}: {exc}", flush=True)
    finally:
        with STATE.lock:
            record.completed_at = time.time()
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


def bootstrap() -> None:
    try:
        # Krea2 is the first image family. Starting it here makes /ready mean the
        # actual Comfy/Krea runtime is available, not merely that the HTTP socket exists.
        adapter = get_adapter("krea2_image")
        adapter.ensure_ready()
        with STATE.lock:
            STATE.worker_status = "idle"
        STATE.mark_idle()
        emit_event("worker_ready", taskFamilies=supported_task_families(), readyAt=time.time())
    except Exception as exc:
        with STATE.lock:
            STATE.worker_status = "unhealthy"
        emit_event("worker_unhealthy", error=str(exc), errorType=type(exc).__name__)
        print(f"[Image Pod] bootstrap failed: {type(exc).__name__}: {exc}", flush=True)


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


class Handler(BaseHTTPRequestHandler):
    server_version = "SceneBuilderImagePod/1.0"

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _authorized(self) -> bool:
        if not POD_TOKEN:
            return True
        return self.headers.get("Authorization", "") == f"Bearer {POD_TOKEN}"

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
            with STATE.lock:
                ready = STATE.worker_status in {"idle", "busy"} and not STATE.draining
                payload = {
                    "ready": ready,
                    "status": STATE.worker_status,
                    "taskFamilies": supported_task_families(),
                }
            self._send_json(HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE, payload)
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
                }
            self._send_json(HTTPStatus.OK, payload)
            return
        if path == "/diagnostics/gpu":
            self._send_json(HTTPStatus.OK, gpu_diagnostics())
            return
        if path.startswith("/jobs/"):
            job_id = path[len("/jobs/") :].strip("/")
            with STATE.lock:
                record = STATE.jobs.get(job_id)
                payload = record.public() if record else None
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
        if path == "/jobs":
            try:
                payload = self._read_json()
                job_id = str(payload.get("jobId") or payload.get("job_id") or "").strip()
                family = str(payload.get("taskFamily") or payload.get("task_family") or "").strip()
                if not job_id:
                    raise ValueError("jobId is required")
                if family not in supported_task_families():
                    raise ValueError(f"unsupported taskFamily: {family or '<missing>'}")

                with STATE.lock:
                    if STATE.draining:
                        self._send_json(HTTPStatus.CONFLICT, {"error": "worker_draining"})
                        return
                    existing = STATE.jobs.get(job_id)
                    if existing:
                        self._send_json(HTTPStatus.OK, existing.public())
                        return
                    if STATE.current_job_id is not None or STATE.worker_status == "busy":
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
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[Image Pod] listening on {HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

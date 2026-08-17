from __future__ import annotations

import gc
import json
import os
import queue
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from src.common.runtime_adapter import run_h3_job


HOST = os.environ.get("H3_POD_HOST", "0.0.0.0")
PORT = int(os.environ.get("H3_POD_PORT", "8000"))
POD_TOKEN = os.environ.get("SCENEBUILDER_POD_TOKEN", "").strip()
WORKER_ID = os.environ.get("SCENEBUILDER_WORKER_ID", "").strip()
CONTROL_URL = os.environ.get("SCENEBUILDER_CONTROL_URL", "").strip()
DEFAULT_IDLE_TIMEOUT = max(0, int(os.environ.get("H3_POD_IDLE_TIMEOUT_SECONDS", "60")))
ROOT = Path(os.environ.get("SCENEBUILDER_H3_ROOT", "/opt/scenebuilder-h3"))
COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))
COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"

ALLOWED_TASK_FAMILIES = {"h3_fl2va", "h3_ref2va"}


@dataclass
class WorkloadRecord:
    workload_id: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    current_job_id: str | None = None
    progress_phase: str = "queued"
    progress_percent: int | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    cancel_requested: bool = False
    cancelled_job_ids: set[str] = field(default_factory=set)

    def public(self) -> dict[str, Any]:
        return {
            "workloadId": self.workload_id,
            "status": self.status,
            "createdAt": self.created_at,
            "startedAt": self.started_at,
            "completedAt": self.completed_at,
            "currentJobId": self.current_job_id,
            "phase": self.progress_phase,
            "progressPercent": self.progress_percent,
            "results": list(self.results),
            "errors": list(self.errors),
            "cancelRequested": self.cancel_requested,
            "cancelledJobIds": sorted(self.cancelled_job_ids),
        }


class PodState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.workloads: dict[str, WorkloadRecord] = {}
        self.queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self.worker_status = "starting"
        self.current_workload_id: str | None = None
        self.loaded_family: str | None = None
        self.idle_since: float | None = None
        self.terminate_after: float | None = None
        self.idle_timeout_seconds = DEFAULT_IDLE_TIMEOUT
        self.draining = False
        self.started_at = time.time()

    def set_ready(self) -> None:
        with self.lock:
            if not self.draining and self.current_workload_id is None:
                self.worker_status = "idle"
                self.idle_since = time.time()
                self.terminate_after = None


STATE = PodState()


def _json_request(url: str, payload: dict[str, Any], timeout: int = 10) -> None:
    if not url:
        return
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "SceneBuilder-H3-Pod/1.0",
    }
    if POD_TOKEN:
        headers["Authorization"] = f"Bearer {POD_TOKEN}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read(1024)
    except Exception as exc:
        print(f"[H3 Pod] control callback failed: {type(exc).__name__}: {exc}", flush=True)


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


def _comfy_post(path: str, payload: dict[str, Any] | None = None, timeout: int = 5) -> bool:
    data = json.dumps(payload or {}, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        f"{COMFY_URL.rstrip('/')}/{path.lstrip('/')}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read(1024)
        return True
    except Exception:
        return False


def release_models_for_family_switch() -> None:
    # ComfyUI supports freeing cached models through /free on current releases.
    # If the endpoint changes, failure is harmless; normal VRAM pressure still
    # lets ComfyUI evict models itself.
    _comfy_post("/free", {"unload_models": True, "free_memory": True})
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def interrupt_current_generation() -> None:
    _comfy_post("/interrupt", {})


def _normalize_idle_timeout(value: Any) -> int:
    if value is None or value == "":
        return DEFAULT_IDLE_TIMEOUT
    try:
        parsed = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("idleTimeoutSeconds must be a non-negative number") from exc
    if parsed < 0:
        raise ValueError("idleTimeoutSeconds must be >= 0")
    return parsed


def _job_id(item: dict[str, Any]) -> str:
    return str(item.get("jobId") or item.get("job_id") or item.get("id") or "unknown")


def _task_family(item: dict[str, Any]) -> str:
    family = str(item.get("taskFamily") or item.get("task_family") or "").strip()
    if family not in ALLOWED_TASK_FAMILIES:
        raise ValueError(f"Unsupported taskFamily: {family or '<missing>'}")
    return family


def _progress_callback(record: WorkloadRecord, job_id: str):
    def callback(phase: str, progress: int | None = None) -> None:
        with STATE.lock:
            record.progress_phase = str(phase)
            record.progress_percent = None if progress is None else max(0, min(100, int(progress)))
        emit_event(
            "job_progress",
            workloadId=record.workload_id,
            jobId=job_id,
            phase=record.progress_phase,
            progressPercent=record.progress_percent,
        )

    return callback


def _finish_worker_idle(record: WorkloadRecord, idle_timeout: int) -> None:
    now = time.time()
    with STATE.lock:
        if STATE.draining:
            return
        STATE.current_workload_id = None
        STATE.worker_status = "idle"
        STATE.idle_since = now
        STATE.idle_timeout_seconds = idle_timeout
        STATE.terminate_after = now + idle_timeout
    emit_event(
        "worker_idle",
        workloadId=record.workload_id,
        idleSince=now,
        idleTimeoutSeconds=idle_timeout,
        terminateAfter=now + idle_timeout,
        loadedFamily=STATE.loaded_family,
    )


def process_workload(payload: dict[str, Any]) -> None:
    workload_id = str(payload.get("workloadId") or payload.get("workload_id") or "").strip()
    with STATE.lock:
        record = STATE.workloads[workload_id]
        record.status = "running"
        record.started_at = time.time()
        record.progress_phase = "preparing_model"
        STATE.worker_status = "busy"
        STATE.current_workload_id = workload_id
        STATE.idle_since = None
        STATE.terminate_after = None

    items = payload.get("items")
    continue_on_failure = bool(payload.get("continueOnItemFailure", True))
    idle_timeout = _normalize_idle_timeout(payload.get("idleTimeoutSeconds"))
    emit_event("workload_started", workloadId=workload_id, itemCount=len(items))

    for index, raw_item in enumerate(items):
        item = dict(raw_item)
        job_id = _job_id(item)
        child_cancelled = False
        with STATE.lock:
            if record.cancel_requested:
                record.status = "cancelled"
                record.progress_phase = "cancelled"
                break
            child_cancelled = job_id in record.cancelled_job_ids
        if child_cancelled:
            emit_event("job_cancelled", workloadId=workload_id, jobId=job_id)
            continue

        try:
            family = _task_family(item)
            if STATE.loaded_family and STATE.loaded_family != family:
                emit_event(
                    "model_switch",
                    workloadId=workload_id,
                    jobId=job_id,
                    fromFamily=STATE.loaded_family,
                    toFamily=family,
                )
                release_models_for_family_switch()

            with STATE.lock:
                record.current_job_id = job_id
                record.progress_phase = "preparing_model"
                record.progress_percent = None

            result = run_h3_job(
                item,
                expected_task_family=family,
                runtime_name="minimax-h3-pod",
                progress_callback=_progress_callback(record, job_id),
            )
            STATE.loaded_family = family
            with STATE.lock:
                child_cancelled = job_id in record.cancelled_job_ids
            if child_cancelled:
                emit_event("job_cancelled", workloadId=workload_id, jobId=job_id)
                continue
            result_summary = {
                "index": index,
                "jobId": job_id,
                "taskFamily": family,
                "ok": bool(result.get("ok", True)),
                "result": result,
            }
            with STATE.lock:
                record.results.append(result_summary)
            emit_event(
                "job_completed",
                workloadId=workload_id,
                jobId=job_id,
                taskFamily=family,
                result=result,
            )
        except Exception as exc:
            with STATE.lock:
                child_cancelled = job_id in record.cancelled_job_ids
            if child_cancelled:
                emit_event("job_cancelled", workloadId=workload_id, jobId=job_id)
                release_models_for_family_switch()
                continue
            error = {
                "index": index,
                "jobId": job_id,
                "error": str(exc),
                "errorType": type(exc).__name__,
            }
            with STATE.lock:
                record.errors.append(error)
                record.progress_phase = "failed"
                record.progress_percent = None
            emit_event("job_failed", workloadId=workload_id, jobId=job_id, **error)
            print(f"[H3 Pod] job {job_id} failed: {type(exc).__name__}: {exc}", flush=True)
            if not continue_on_failure:
                with STATE.lock:
                    record.status = "failed"
                break
            release_models_for_family_switch()

    with STATE.lock:
        if record.status == "running":
            if record.errors and record.results:
                record.status = "partial_failed"
            elif record.errors:
                record.status = "failed"
            else:
                record.status = "completed"
            record.progress_phase = record.status if record.status != "partial_failed" else "completed"
            record.progress_percent = 100 if record.results else None
        record.current_job_id = None
        record.completed_at = time.time()
        public = record.public()

    emit_event("workload_completed", workloadId=workload_id, workload=public)
    _finish_worker_idle(record, idle_timeout)


def worker_loop() -> None:
    STATE.set_ready()
    emit_event("worker_ready", readyAt=time.time())
    while True:
        payload = STATE.queue.get()
        try:
            process_workload(payload)
        except Exception as exc:
            workload_id = str(payload.get("workloadId") or payload.get("workload_id") or "")
            print(f"[H3 Pod] workload {workload_id} crashed: {type(exc).__name__}: {exc}", flush=True)
            with STATE.lock:
                record = STATE.workloads.get(workload_id)
                if record:
                    record.status = "failed"
                    record.progress_phase = "failed"
                    record.errors.append({"error": str(exc), "errorType": type(exc).__name__})
                    record.completed_at = time.time()
                STATE.current_workload_id = None
                STATE.worker_status = "idle"
                STATE.idle_since = time.time()
                STATE.terminate_after = STATE.idle_since + DEFAULT_IDLE_TIMEOUT
            emit_event("workload_failed", workloadId=workload_id, error=str(exc), errorType=type(exc).__name__)
        finally:
            STATE.queue.task_done()


def idle_watchdog() -> None:
    while True:
        time.sleep(1)
        should_expire = False
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
                should_expire = True
        if should_expire:
            emit_event("idle_expired", terminateAfter=deadline, loadedFamily=STATE.loaded_family)
            # Provider deletion belongs to the SceneBuilder control plane. This
            # process stays alive long enough for a lost callback to be recovered
            # by the D1/provider reaper, but it refuses new workloads while draining.


def _required_files() -> dict[str, bool]:
    files = {
        "fl2vaModel": COMFY_ROOT / "models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "ref2vaModel": COMFY_ROOT / "models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "videoVae": COMFY_ROOT / "models/vae/minimax_h3_video_vae_fp16.safetensors",
        "audioVae": COMFY_ROOT / "models/vae/minimax_h3_audio_vae_fp32.safetensors",
        "fl2vaWorkflow": ROOT / "workflows/fl2va_master.json",
        "ref2vaWorkflow": ROOT / "workflows/ref2va_master.json",
        "fl2vaManifest": ROOT / "workflows/manifests/fl2va_manifest.json",
        "ref2vaManifest": ROOT / "workflows/manifests/ref2va_manifest.json",
    }
    return {key: path.is_file() for key, path in files.items()}


def _cuda_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "available": False,
        "deviceCount": 0,
    }
    try:
        import torch

        summary.update(
            {
                "torchVersion": getattr(torch, "__version__", None),
                "torchCudaVersion": getattr(torch.version, "cuda", None),
                "available": bool(torch.cuda.is_available()),
                "deviceCount": int(torch.cuda.device_count()),
            }
        )
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            summary["deviceName"] = torch.cuda.get_device_name(0)
            summary["capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    return summary


def _nvidia_smi_summary() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
        rows = []
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 7:
                rows.append(
                    {
                        "name": parts[0],
                        "uuid": parts[1],
                        "driverVersion": parts[2],
                        "memoryTotalMiB": parts[3],
                        "memoryUsedMiB": parts[4],
                        "memoryFreeMiB": parts[5],
                        "utilizationGpuPercent": parts[6],
                    }
                )
        return {"ok": True, "gpus": rows}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "gpus": []}


def diagnostics() -> dict[str, Any]:
    disk = subprocess.run(["df", "-Pk", "/"], capture_output=True, text=True, timeout=5)
    with STATE.lock:
        worker = {
            "workerId": WORKER_ID or None,
            "status": STATE.worker_status,
            "currentWorkloadId": STATE.current_workload_id,
            "loadedFamily": STATE.loaded_family,
            "idleSince": STATE.idle_since,
            "terminateAfter": STATE.terminate_after,
            "idleTimeoutSeconds": STATE.idle_timeout_seconds,
            "uptimeSeconds": round(time.time() - STATE.started_at, 3),
        }
    return {
        "ok": True,
        "worker": worker,
        "nvidiaSmi": _nvidia_smi_summary(),
        "cuda": _cuda_summary(),
        "files": _required_files(),
        "comfy": {"host": COMFY_HOST, "port": COMFY_PORT},
        "diskPk": disk.stdout.strip() if disk.returncode == 0 else None,
    }


def ready_payload() -> tuple[bool, dict[str, Any]]:
    files = _required_files()
    cuda = _cuda_summary()
    ready = all(files.values()) and bool(cuda.get("available")) and int(cuda.get("deviceCount") or 0) >= 1
    with STATE.lock:
        if STATE.draining:
            ready = False
        status = STATE.worker_status
    return ready, {
        "ok": ready,
        "status": status,
        "cuda": {"available": cuda.get("available"), "deviceCount": cuda.get("deviceCount"), "deviceName": cuda.get("deviceName")},
        "files": files,
    }


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "SceneBuilderH3Pod/1.0"

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        if not POD_TOKEN:
            self._send(HTTPStatus.SERVICE_UNAVAILABLE, {"ok": False, "error": "Pod auth token is not configured"})
            return False
        supplied = self.headers.get("Authorization", "")
        if supplied != f"Bearer {POD_TOKEN}":
            self._send(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "Unauthorized"})
            return False
        return True

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        if length > 16 * 1024 * 1024:
            raise ValueError("Request body too large")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send(HTTPStatus.OK, {"ok": True, "runtime": "minimax-h3-pod"})
            return
        if path == "/ready":
            ready, payload = ready_payload()
            self._send(HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE, payload)
            return
        if path in {"/diagnostics", "/diagnostics/gpu"}:
            if not self._authorized():
                return
            payload = diagnostics()
            if path == "/diagnostics/gpu":
                payload = {"ok": True, "nvidiaSmi": payload["nvidiaSmi"], "cuda": payload["cuda"]}
            self._send(HTTPStatus.OK, payload)
            return
        if path.startswith("/workloads/"):
            if not self._authorized():
                return
            workload_id = path[len("/workloads/") :].strip("/")
            with STATE.lock:
                record = STATE.workloads.get(workload_id)
            if not record:
                self._send(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Unknown workload"})
                return
            self._send(HTTPStatus.OK, {"ok": True, "workload": record.public()})
            return
        self._send(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if not self._authorized():
            return
        try:
            payload = self._read_json()
        except Exception as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if path == "/workloads":
            workload_id = str(payload.get("workloadId") or payload.get("workload_id") or "").strip()
            items = payload.get("items")
            if not workload_id:
                self._send(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "workloadId is required"})
                return
            if not isinstance(items, list) or not items or not all(isinstance(item, dict) for item in items):
                self._send(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "items must be a non-empty array of objects"})
                return
            try:
                idle_timeout = _normalize_idle_timeout(payload.get("idleTimeoutSeconds"))
                for item in items:
                    _task_family(item)
            except ValueError as exc:
                self._send(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return

            with STATE.lock:
                if STATE.draining:
                    self._send(HTTPStatus.CONFLICT, {"ok": False, "error": "Worker is draining"})
                    return
                existing = STATE.workloads.get(workload_id)
                if existing:
                    self._send(HTTPStatus.OK, {"ok": True, "accepted": True, "duplicate": True, "workload": existing.public()})
                    return
                if STATE.current_workload_id is not None or not STATE.queue.empty():
                    self._send(HTTPStatus.CONFLICT, {"ok": False, "error": "Worker is busy"})
                    return
                record = WorkloadRecord(workload_id=workload_id)
                STATE.workloads[workload_id] = record
                STATE.worker_status = "busy"
                STATE.current_workload_id = workload_id
                STATE.idle_since = None
                STATE.terminate_after = None
                STATE.idle_timeout_seconds = idle_timeout
                STATE.queue.put_nowait(payload)
            emit_event("workload_accepted", workloadId=workload_id, itemCount=len(items), idleTimeoutSeconds=idle_timeout)
            self._send(HTTPStatus.ACCEPTED, {"ok": True, "accepted": True, "workloadId": workload_id})
            return

        parts = [part for part in path.split("/") if part]
        if len(parts) == 5 and parts[0] == "workloads" and parts[2] == "items" and parts[4] == "cancel":
            workload_id = parts[1]
            job_id = parts[3]
            with STATE.lock:
                record = STATE.workloads.get(workload_id)
                if not record:
                    self._send(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Unknown workload"})
                    return
                record.cancelled_job_ids.add(job_id)
                active = record.current_job_id == job_id
            if active:
                interrupt_current_generation()
            emit_event("job_cancel_requested", workloadId=workload_id, jobId=job_id)
            self._send(HTTPStatus.ACCEPTED, {"ok": True, "jobId": job_id, "cancelRequested": True})
            return

        if path.startswith("/workloads/") and path.endswith("/cancel"):
            workload_id = path[len("/workloads/") : -len("/cancel")].strip("/")
            with STATE.lock:
                record = STATE.workloads.get(workload_id)
                if not record:
                    self._send(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Unknown workload"})
                    return
                record.cancel_requested = True
                active = record.current_job_id is not None
            if active:
                interrupt_current_generation()
            emit_event("workload_cancel_requested", workloadId=workload_id)
            self._send(HTTPStatus.ACCEPTED, {"ok": True, "cancelRequested": True})
            return

        self._send(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[H3 Pod HTTP] {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    if not POD_TOKEN:
        print("[H3 Pod] WARNING: SCENEBUILDER_POD_TOKEN is missing; protected endpoints will reject requests", flush=True)
    threading.Thread(target=worker_loop, name="h3-pod-worker", daemon=True).start()
    threading.Thread(target=idle_watchdog, name="h3-pod-idle-watchdog", daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), RequestHandler)
    print(f"[H3 Pod] minimax-h3-pod listening on {HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

from __future__ import annotations

import os

from . import server
from .engine_runtime import is_fatal_cuda_error

_original_error_code = server._error_code
_original_event = server._event
_FATAL_CUDA_CONTEXT = False


def _guarded_error_code(error: BaseException) -> str:
    if is_fatal_cuda_error(error):
        return "CUDA_ILLEGAL_ADDRESS"
    return _original_error_code(error)


def _guarded_event(event_type: str, **extra):
    global _FATAL_CUDA_CONTEXT
    if event_type == "job_failed" and str(extra.get("errorCode") or "") == "CUDA_ILLEGAL_ADDRESS":
        _FATAL_CUDA_CONTEXT = True
        # CUDA illegal-address/launch failures can poison the process context.
        # Stop advertising readiness immediately; the control plane will reap
        # this worker instead of handing it another job.
        server._READY = False
        server._DRAINING = True
        server._STARTUP_ERROR = "CUDA context quarantined after illegal memory access"
        return _original_event(event_type, **extra)
    if event_type == "worker_idle" and _FATAL_CUDA_CONTEXT:
        return _original_event(
            "worker_unhealthy",
            errorCode="CUDA_ILLEGAL_ADDRESS",
            error="CUDA context quarantined after illegal memory access; replace this pod before more work",
        )
    return _original_event(event_type, **extra)


server._error_code = _guarded_error_code
server._event = _guarded_event
app = server.app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("H3_POD_PORT", "8000")))

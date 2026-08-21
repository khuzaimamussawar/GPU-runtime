from __future__ import annotations

import os
from dataclasses import dataclass


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class RuntimeConfig:
    worker_id: str
    pod_token: str
    control_url: str
    r2_bucket: str
    r2_endpoint: str
    r2_access_key: str
    r2_secret_key: str
    r2_region: str
    r2_public_url: str
    port: int
    idle_timeout_seconds: int
    service_kind: str
    debug: bool

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        # These are intentionally the same names already used by the H3 pod
        # runtime. Do not add ENHANCER_* duplicates for shared concepts.
        return cls(
            worker_id=_required("SCENEBUILDER_WORKER_ID"),
            pod_token=_required("SCENEBUILDER_POD_TOKEN"),
            control_url=_required("SCENEBUILDER_CONTROL_URL").rstrip("/"),
            r2_bucket=_required("R2_BUCKET_NAME"),
            r2_endpoint=_required("R2_ENDPOINT"),
            r2_access_key=_required("R2_ACCESS_KEY"),
            r2_secret_key=_required("R2_SECRET_KEY"),
            r2_region=os.environ.get("R2_REGION", "auto").strip() or "auto",
            r2_public_url=os.environ.get("R2_PUBLIC_URL", "").strip().rstrip("/"),
            port=_int("H3_POD_PORT", 8000, 1),
            idle_timeout_seconds=_int("H3_POD_IDLE_TIMEOUT_SECONDS", 60, 0),
            service_kind=os.environ.get("SCENEBUILDER_SERVICE_KIND", "enhancer").strip() or "enhancer",
            debug=os.environ.get("SCENEBUILDER_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"},
        )

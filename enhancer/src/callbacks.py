from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
import urllib.request
from typing import Any

from .config import RuntimeConfig

EVENT_PATH = "/api/projects/v2/enhancer/pod/events"


def canonical_body(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign_event(token: str, worker_id: str, timestamp_ms: int, nonce: str, body: bytes) -> str:
    message = b"\n".join([
        b"enhancer-event-v1",
        worker_id.encode("utf-8"),
        str(timestamp_ms).encode("ascii"),
        nonce.encode("ascii"),
        body,
    ])
    return hmac.new(token.encode("utf-8"), message, hashlib.sha256).hexdigest()


def post_event(config: RuntimeConfig, payload: dict[str, Any], timeout: float = 15.0) -> dict[str, Any] | None:
    timestamp_ms = int(time.time() * 1000)
    nonce = secrets.token_hex(16)
    body = canonical_body(payload)
    signature = sign_event(config.pod_token, config.worker_id, timestamp_ms, nonce, body)
    request = urllib.request.Request(
        f"{config.control_url}{EVENT_PATH}",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-SceneBuilder-Worker-Id": config.worker_id,
            "X-SceneBuilder-Timestamp": str(timestamp_ms),
            "X-SceneBuilder-Nonce": nonce,
            "X-SceneBuilder-Signature": signature,
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        if not raw:
            return None
        return json.loads(raw)

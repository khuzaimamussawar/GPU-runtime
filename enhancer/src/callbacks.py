from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .config import RuntimeConfig

EVENT_PATH = "/api/projects/v2/enhancer/pod/events"
H3_EVENT_PATH = "/api/projects/v2/h3/pod/events"


def event_url(control_url: str) -> str:
    url = control_url.rstrip("/")
    if url.endswith(EVENT_PATH) or url.endswith(H3_EVENT_PATH):
        return url
    return f"{url}{EVENT_PATH}"


def post_event(config: RuntimeConfig, payload: dict[str, Any], timeout: float = 15.0) -> dict[str, Any] | None:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    urls = [config.control_url]
    if config.fallback_control_url and config.fallback_control_url not in urls:
        urls.append(config.fallback_control_url)
    last_error: Exception | None = None
    for index, control_url in enumerate(urls):
        try:
            return _post_event_once(config, control_url, body, timeout)
        except RuntimeError as error:
            last_error = error
            if index >= len(urls) - 1 or not _should_try_fallback(error):
                raise
    if last_error:
        raise last_error
    return None


def _should_try_fallback(error: Exception) -> bool:
    text = str(error)
    return text.startswith("HTTP 401 ") or text.startswith("HTTP 403 ") or text.startswith("HTTP 200 NON_JSON ")


def _post_event_once(
    config: RuntimeConfig,
    control_url: str,
    body: bytes,
    timeout: float,
) -> dict[str, Any] | None:
    request = urllib.request.Request(
        event_url(control_url),
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "SceneBuilder-Enhancer-Pod/1.0",
            "Authorization": f"Bearer {config.pod_token}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            if not raw:
                return None
            try:
                return json.loads(raw)
            except json.JSONDecodeError as error:
                text = raw.decode("utf-8", "replace")[:1000]
                raise RuntimeError(f"HTTP 200 NON_JSON from {control_url}: {text}") from error
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")[:1000]
        raise RuntimeError(f"HTTP {error.code} {error.reason} from {control_url}: {body}") from error

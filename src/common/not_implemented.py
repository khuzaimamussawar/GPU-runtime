from __future__ import annotations

import json


def fail_until_comfy_runtime_is_wired(runtime: str) -> None:
    payload = {
        "status": "not_implemented",
        "runtime": runtime,
        "message": "Docker/build plumbing is present. Comfy execution adapter is not wired yet.",
    }
    print(json.dumps(payload), flush=True)
    raise SystemExit(64)

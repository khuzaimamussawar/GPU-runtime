from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class H3Job:
    job_id: str
    task_family: str
    mode: str
    prompt: str
    width: int
    height: int
    duration_seconds: float
    fps: int = 24
    settings: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return max(1, round(self.duration_seconds * self.fps))


def normalize_job(payload: dict[str, Any]) -> H3Job:
    return H3Job(
        job_id=str(payload["jobId"]),
        task_family=str(payload["taskFamily"]),
        mode=str(payload["mode"]),
        prompt=str(payload.get("prompt", "")),
        width=int(payload["width"]),
        height=int(payload["height"]),
        duration_seconds=float(payload["durationSeconds"]),
        fps=int(payload.get("fps", 24)),
        settings=dict(payload.get("settings") or {}),
        inputs=dict(payload.get("inputs") or {}),
    )

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class H3Job:
    job_id: str
    project_id: str
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
    job_id = payload.get("jobId") or payload.get("job_id") or payload.get("id")
    if not job_id:
        raise ValueError("H3 job payload requires jobId")
    project_id = (
        payload.get("projectId")
        or payload.get("project_id")
        or payload.get("project")
        or "unknown-project"
    )
    return H3Job(
        job_id=str(job_id),
        project_id=str(project_id),
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

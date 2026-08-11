import json

import runpod

from src.common.runtime_adapter import run_h3_job


def _progress_reporter(job):
    def report(phase, progress=None):
        update = {"phase": str(phase)}
        if progress is not None:
            update["progress"] = int(progress)
        runpod.serverless.progress_update(job, json.dumps(update, separators=(",", ":")))

    return report


def handler(job):
    payload = dict(job.get("input") or {})
    payload.setdefault("jobId", job.get("id"))
    return run_h3_job(
        payload,
        "h3_fl2va",
        "runpod-fl2va",
        progress_callback=_progress_reporter(job),
    )


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

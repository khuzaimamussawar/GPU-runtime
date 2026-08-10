from src.common.h3_runtime import run_h3_job


def handler(job):
    payload = dict(job.get("input") or {})
    payload.setdefault("jobId", job.get("id"))
    return run_h3_job(payload, "h3_fl2va", "runpod-fl2va")


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})

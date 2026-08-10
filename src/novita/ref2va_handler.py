from src.common.runtime_adapter import run_h3_job


def handler(job):
    payload = dict(job.get("input") or {})
    payload.setdefault("jobId", job.get("id"))
    return run_h3_job(payload, "h3_ref2va", "novita-ref2va")


if __name__ == "__main__":
    # Novita Async Serverless accepts RunPod-compatible queue workers; its
    # official ComfyUI example is also a queue-worker image.
    import runpod

    runpod.serverless.start({"handler": handler})

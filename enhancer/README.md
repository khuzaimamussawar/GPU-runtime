# SceneBuilder Enhancer GPU Runtime

This directory is deliberately isolated from the existing H3 `src/`, `docker/`, `workflows/`, and build scripts.

- `enhancer_fast`: CUDA 13.0/cu130, Real-ESRGAN image upscaling and FAST video ESRGAN/RIFE runtime. This is the runtime used by Storyboard **Upscale All** and video-only ESRGAN/RIFE TensorRT engine builder jobs.
- `enhancer_quality`: CUDA 13.0/cu130, official FlashVSR v1.1 pipeline for video super-resolution on its known-compatible GPU classes.

Both expose the same Pod API: public `/health`, `/ready`, `/capabilities`, and bearer-protected `/jobs` plus cancellation/status. The controller supplies R2 credentials, worker-scoped HMAC token, callback URL, worker ID, service kind, debug flag, and idle timeout at provider launch.

## Hetzner Docker build aliases

The enhancer workflow builds two final runtime images, but does so through reusable parent layers:

```text
smoke -> base -> torch -> vfi-models -> esrgan-models -> fast
                         -> flashvsr-runtime -> flashvsr-models -> quality
```

Use `.github/workflows/hetzner-enhancer-build.yml` on `cpx32` by default.

- `target=all` expands to `smoke,base,torch,vfi-models,esrgan-models,fast,flashvsr-runtime,flashvsr-models,quality`.
- `target=remaining` checks Docker Hub for already-pushed `scenebuilder-enhancer-<target>:<image_tag>` images and builds only missing layers. Pass `previous_workflow_run_id` for traceability to the failed/partial run you are continuing.
- `targets_csv` still accepts explicit comma-separated targets and can include `all` or `remaining`.

# SceneBuilder Enhancer GPU Runtime

This directory is deliberately isolated from the existing H3 `src/`, `docker/`, `workflows/`, and build scripts.

- `enhancer_fast`: CUDA 13.0/cu130, Real-ESRGAN image upscaling and FAST video ESRGAN/RIFE runtime. This is the runtime used by Storyboard **Upscale All** and video-only ESRGAN/RIFE TensorRT engine builder jobs.
- `enhancer_quality`: CUDA 13.0/cu130, official FlashVSR v1.1 pipeline for video super-resolution on its known-compatible GPU classes.

Both expose the same Pod API: public `/health`, `/ready`, `/capabilities`, and bearer-protected `/jobs` plus cancellation/status. The controller supplies R2 credentials, worker-scoped HMAC token, callback URL, worker ID, service kind, debug flag, and idle timeout at provider launch.

## Hetzner Docker build aliases

The enhancer workflow builds two separate final runtime images through a shared base and two branches:

```text
smoke
-> base
   -> torch
      -> vfi-models
         -> FAST branch: esrgan-models -> fast
         -> QUALITY branch: flashvsr-runtime -> flashvsr-models -> quality
```

Use `.github/workflows/hetzner-enhancer-build.yml` on `cpx32` by default.

- `target=fast` expands to `smoke,base,torch,vfi-models,esrgan-models,fast`.
- `target=quality` expands to `smoke,base,torch,vfi-models,flashvsr-runtime,flashvsr-models,quality`.
- `target=all` expands to `smoke,base,torch,vfi-models,esrgan-models,fast,flashvsr-runtime,flashvsr-models,quality`.
- `target=remaining` checks Docker Hub for already-pushed `scenebuilder-enhancer-<target>:<image_tag>` images and builds only missing layers. Pass `previous_workflow_run_id` for traceability to the failed/partial run you are continuing.
- `targets_csv` still accepts explicit comma-separated targets and can include `all`, `remaining`, `fast`, or `quality`. Duplicated shared layers are skipped inside a single run.

Docker builds do not create TensorRT `.engine` files. FAST carries the engine-builder runtime path for video-only ESRGAN/RIFE jobs; generated engines are built later on compatible NVIDIA GPU pods and stored in private R2/D1. QUALITY is a separate runtime image and may consume compatible active RIFE engines, but it is not the engine-builder image.

The model layers bake the three approved ONNX builder inputs with SHA-256 verification:

- `realesr-animevideov3.onnx` from `tidus2102/Real-ESRGAN`, SHA-256 `00ece3ac21c43ee31459216b5174b2cea0c5325044c5142aeb840f4890e175ff`.
- `realesr-general-x4v3.onnx` from `CoderViking/realesr-general-x4v3-onnx`, SHA-256 `1940a93ee08283a0a7286183186357b1688fe9fa8ede74604b424586aaddf112`.
- `rife-4.9.onnx` from `yuvraj108c/rife-onnx`, SHA-256 `76e4cef9ab42fa7dd4e8f6e4aba47462051e3faa969e4bca6479784fbab0ac6f`.

These ONNX files are build inputs only. Product inference uses active `.engine` files when available; ONNX is needed only to build/rebuild engines or to validate provenance.

# SceneBuilder Enhancer GPU Runtime

This directory is deliberately isolated from the existing H3 `src/`, `docker/`, `workflows/`, and build scripts.

- `enhancer_fast`: CUDA 12.8/cu128, Real-ESRGAN image upscaling. This is the runtime used by Storyboard **Upscale All**.
- `enhancer_quality`: CUDA 12.4/cu124, official FlashVSR v1.1 pipeline for video super-resolution on its known-compatible GPU classes.

Both expose the same Pod API: public `/health`, `/ready`, `/capabilities`, and bearer-protected `/jobs` plus cancellation/status. The controller supplies R2 credentials, worker-scoped HMAC token, callback URL, worker ID, service kind, debug flag, and idle timeout at provider launch.

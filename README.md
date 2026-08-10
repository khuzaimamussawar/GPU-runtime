# MiniMax H3 Serverless

Layered Docker build system for SceneBuilder MiniMax H3 FL2VA and Ref2VA runtimes.

This repo owns:

- Dockerfiles for CUDA/PyTorch/Comfy/H3 layers
- RunPod and Novita handler entrypoints
- Comfy workflow JSON and manifests
- FFmpeg output contract scripts
- Hetzner temporary builder orchestration

SceneBuilder owns:

- UI
- D1/R2 orchestration
- endpoint registry
- job routing

## First Safe Test

Run the GitHub Actions workflow `Hetzner Docker Build` with:

```text
target: smoke
server_type: ccx43
debug_keep_server_on_failure: false
```

This proves:

```text
GitHub Actions -> temporary Hetzner server -> Docker buildx --push -> Docker Hub -> delete server
```

without downloading H3 model files.

## One-Server Batch Builds

Hetzner bills by the hour, so prefer batching dependent layers into one workflow run instead of booting one server per layer.

Use `targets_csv` to build several targets sequentially on the same temporary server:

```text
target: smoke
targets_csv: qwen-nvfp4,qwen-all
server_type: ccx33
delete_server_after_success: true
debug_keep_server_on_failure: false
```

When `targets_csv` is set, `target` is only a harmless required dropdown value. The comma-separated batch is what actually runs.

Good batches:

```text
qwen-nvfp4,qwen-all
fl2va-base,fl2va-workflow,fl2va-loras,runpod-fl2va,novita-fl2va
ref2va-base,ref2va-workflow,ref2va-loras,runpod-ref2va,novita-ref2va
```

After `qwen-all` exists, this builds the four deployable endpoint images in one server boot:

```text
target: smoke
targets_csv: fl2va-base,fl2va-workflow,fl2va-loras,runpod-fl2va,novita-fl2va,ref2va-base,ref2va-workflow,ref2va-loras,runpod-ref2va,novita-ref2va
server_type: ccx33
delete_server_after_success: true
```

The remote script removes the previous checkout before cloning the current repo state, but it keeps Docker layer cache/images on the same builder during that workflow run. That is intentional: it avoids redownloading/rebuilding parent layers.

`delete_server_after_success: false` keeps the Hetzner server alive after success. Use it only when you are watching the clock and will delete the server manually, because billing continues until deletion.

## Required GitHub Secrets

```text
HETZNER_TOKEN
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
HF_TOKEN
```

Optional if this repo becomes private and Hetzner clones it directly:

```text
GH_FH_TOKEN_MM_H3_SERVERLESS
```

## Runtime Stack Target

```text
CUDA: 13.0 / cu130
Python: 3.13 target, 3.12 fallback only if a required node fails reproducibly
PyTorch: 2.13
Attention: SageAttention 2.2
```

Do not switch to CUDA 13.2 unless CUDA 13.0/cu130 fails reproducibly with the required PyTorch/SageAttention/Comfy nodes.

The first heavy build should prove the base image before downloading H3 weights:

```text
target: base
```

If the selected NVIDIA CUDA image tag is unavailable, change only `CUDA_IMAGE` in `docker/Dockerfile.base` to another CUDA 13.0 cu130-compatible tag and rerun the base build. Do not jump to CUDA 13.2 for a tag-name issue.

## Image Repos

Final runtime images:

```text
<dockerhub-user>/minimax-h3-runpod-fl2va:latest
<dockerhub-user>/minimax-h3-runpod-ref2va:latest
<dockerhub-user>/minimax-h3-novita-fl2va:latest
<dockerhub-user>/minimax-h3-novita-ref2va:latest
```

Heavy layers are shared by Docker content digest where possible.

## Option A Encoder Layout

SceneBuilder uses 4 final endpoint images, not 8. Each final image contains both Qwen text encoders, while the encoder layers remain separate:

```text
base
qwen-nvfp4
qwen-all     adds qwen-int8 on top of qwen-nvfp4
```

`qwen-nvfp4` stays reusable as its own Docker image/layer. `qwen-all` does not redownload NVFP4; it inherits the already-built NVFP4 layer and adds `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`.

The frontend sends the selected encoder, and the runtime handler swaps the Comfy `clip_name` field from the workflow manifest before inference.

Build order:

```text
base
qwen-nvfp4
qwen-all
fl2va-base
ref2va-base
fl2va-workflow
ref2va-workflow
fl2va-loras
ref2va-loras
runpod-fl2va
runpod-ref2va
novita-fl2va
novita-ref2va
```

`qwen-int8` remains available as an optional diagnostic target. The final 4-image deployment path uses `qwen-all` so each runtime can switch between NVFP4 and INT8 from the request payload.

## Runtime Handler Contract

RunPod uses the normal serverless job `input` object. Novita uses an HTTP JSON body with the same shape.

Minimum FL2VA I2V payload:

```json
{
  "jobId": "job_123",
  "projectId": "proj_123",
  "taskFamily": "h3_fl2va",
  "mode": "i2v",
  "prompt": "Provider-facing video prompt.",
  "width": 1056,
  "height": 608,
  "durationSeconds": 4.2,
  "fps": 24,
  "settings": {
    "textEncoder": "nvfp4",
    "steps": 20,
    "sampler": "uni_pc",
    "scheduler": "simple",
    "seed": 12345,
    "spectrum": {
      "enabled": false
    }
  },
  "inputs": {
    "firstFrame": {
      "objectKey": "projects/proj_123/scene_images/original/scene_001.png"
    },
    "outputPrefix": "projects/proj_123/scene_videos"
  }
}
```

Minimum Ref2VA payload:

```json
{
  "jobId": "job_456",
  "projectId": "proj_123",
  "taskFamily": "h3_ref2va",
  "mode": "r2v",
  "prompt": "Use these ordered references as visual guidance.",
  "width": 1056,
  "height": 608,
  "durationSeconds": 6,
  "fps": 24,
  "settings": {
    "textEncoder": "int8",
    "referenceFit": "max",
    "steps": 20,
    "sageAttention": true
  },
  "inputs": {
    "referenceImages": [
      {"objectKey": "projects/proj_123/scene_images/original/scene_003.png"},
      {"objectKey": "projects/proj_123/scene_images/original/scene_004.png"}
    ],
    "referenceAudio": {
      "objectKey": "temp/video-audio/job_456.wav"
    },
    "outputPrefix": "projects/proj_123/scene_videos"
  }
}
```

Returned output fields:

```json
{
  "ok": true,
  "outputs": {
    "master": {
      "objectKey": "projects/proj_123/scene_videos/original/job_456_h265.mp4",
      "url": "https://...",
      "uploaded": true
    },
    "preview": {
      "objectKey": "projects/proj_123/scene_videos/preview/job_456_h264_preview.mp4",
      "url": "https://...",
      "uploaded": true
    }
  }
}
```

SceneBuilder should save object keys in D1 timeline/job rows and derive public URLs only when rendering UI previews.

## H3 Image Preparation Contract

SceneBuilder should send exact `width` and `height` values resolved from the project aspect ratio and MP preset. The H3 runtime preprocesses image inputs before passing them to ComfyUI:

- FL2VA first/last-frame images always use `match`.
- `match` means: largest possible center crop to the target aspect ratio, then Lanczos resize to the exact requested H3 `width x height`.
- `max` means: preserve the whole image, resize down only if the longest side exceeds `referenceMaxSize` / `maxInputSize` / `1024`, and do not upscale.
- `contain` means: preserve the whole image and pad to the exact requested H3 frame size.

Normal users should see friendly labels such as `Match project frame`, `Fit full image`, or `Auto`. Do not expose raw Comfy node names. Advanced/admin UI can still send:

```json
{
  "settings": {
    "referenceFit": "max",
    "referenceMaxSize": 1024
  }
}
```

For keyframes/scene images, use `match`. For character, object, style, mood, or loose visual references, use `max` unless the user explicitly wants them framed as full video panels.

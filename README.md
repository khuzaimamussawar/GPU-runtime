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

SceneBuilder uses 4 final endpoint images, not 8. Each final image contains both Qwen text encoders:

```text
qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
qwen3vl_32b_minimax_h3_int8_convrot.safetensors
```

The frontend sends the selected encoder, and the runtime handler swaps the Comfy `clip_name` field from the workflow manifest before inference.

Build order:

```text
base
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

`qwen-nvfp4` and `qwen-int8` remain available only as optional diagnostic targets. They are not used by the final 4-image deployment path.

# SceneBuilder Enhancer Implementation Plan

Status: **canonical implementation plan**

Branch: `feat/enhancer-gpu-runtime`

This file is intentionally separate from the H3 runtime and H3 lifecycle design. Do not modify H3 lifecycle behavior to implement the enhancer.

---

## 1. Goal

Build a SceneBuilder enhancer runtime that behaves operationally like the existing H3 pod system while remaining physically isolated from H3 state and preserving existing SceneBuilder D1/R2 product contracts.

The enhancer supports:

- Storyboard image upscale.
- Director video upscale.
- Video frame interpolation.
- FAST and QUALITY GPU runtimes.
- RunPod and Novita provisioning.
- H3-style priority, retries, debug/error state, heartbeats, idle reuse, idle timeout, job timeout, provider fallback, delete locks, delete verification, and orphan cleanup.

The enhancer must never silently run neural inference on CPU.

---

## 2. Hard non-negotiables

### 2.1 H3 lifecycle must remain untouched

H3 state:

```text
h3_pod_workers
video_generation_batches
video_generation_jobs
```

Enhancer state:

```text
enhancer_pod_workers
pending_upscales
enhancer_pod_delete_locks
enhancer_event_nonces
enhancer_config
```

Rules:

- Never insert enhancer rows into `h3_pod_workers`.
- H3 code never queries `enhancer_pod_workers`.
- Enhancer code never treats `h3_pod_workers` as its worker pool.
- Keep separate enhancer delete locks.
- Keep a separate enhancer pod callback endpoint.
- Use a separate HMAC derivation domain for enhancer pod tokens.
- Preserve the existing H3 pod lifecycle implementation files.

### 2.2 Existing product tables stay unchanged

Do not add enhancer-specific product schema to:

```text
projects_timeline
project_video_timeline
```

Do not add a second project media state table.

Operational D1 additions are allowed only for enhancer worker/job infrastructure.

### 2.3 Existing R2 layout stays intact

Permanent output paths:

```text
Storyboard image upscale:
projects/{projectId}/images/scenes/upscaled/

Director video upscale:
projects/{projectId}/video/upscaled/
```

Use the existing R2 bucket/credentials and product writeback contract.

### 2.4 Replicate routing

- Storyboard **Upscale All** must use enhancer pods only.
- No Replicate code path may claim enhancer batch jobs.
- Existing single-image `/api/upscale` remains the legacy Replicate path unless explicitly changed later.

---

## 3. Final runtime topology

There are exactly two enhancer GPU image families:

```text
scenebuilder-enhancer-fast
scenebuilder-enhancer-quality
```

**FILM is not part of the product or runtime. Do not install, expose, enqueue, advertise, or route FILM anywhere.**

### 3.1 FAST image

```text
CUDA 13.x userspace
PyTorch CUDA-13 build
CuPy CUDA-13 package
TensorRT 10.16.1 runtime/builder
FFmpeg + NVDEC/NVENC

Real-ESRGAN image models:
  1. RealESRGAN_x4plus_anime_6B
  2. RealESRGAN_x4plus

Real-ESRGAN video models:
  3. realesr-animevideov3
  4. realesr-general-x4v3

VFI:
  5. RIFE
  6. GIMM-VFI-F
```

### 3.2 QUALITY image

```text
CUDA 13.x userspace
PyTorch CUDA-13 build
CuPy CUDA-13 package
TensorRT 10.16.1 runtime/builder for RIFE engine execution
FFmpeg + NVDEC/NVENC

Video upscale:
  1. FlashVSR

VFI:
  2. RIFE
  3. GIMM-VFI-F
```

### 3.3 User-facing VFI choices

The VFI choices are:

```text
RIFE       -> fast path; TensorRT engine where validated
GIMM-VFI-F -> quality/slower path; native PyTorch + CuPy
```

Use **GIMM-VFI-F**, not GIMM-VFI-R and not the perceptual `F-P` variant, unless product requirements deliberately change later.

Reference comparison:
https://www.reddit.com/r/StableDiffusion/comments/1j2evqn/wan_14b_with_mmaudio_gimmvfif_frame_interpolation/

---

## 4. CUDA / PyTorch / CuPy baseline

### 4.1 CUDA baseline

Use CUDA 13 as the enhancer baseline.

Target implementation baseline:

```text
CUDA userspace: 13.x
PyTorch: CUDA-13 build validated with the selected image
CuPy: cupy-cuda13x
TensorRT: 10.16.1
NVIDIA driver: CUDA-13-capable provider driver
```

Do not install multiple CUDA major lines in one image.

Never build this:

```text
CUDA 11 + CUDA 12.8 + CUDA 13
```

There is one system CUDA userspace line per Docker image.

### 4.2 CuPy

Install only:

```text
cupy-cuda13x
```

Do not install `cupy-cuda12x` beside it.

The runtime owns CuPy installation directly. Third-party installers must not downgrade or double-install CuPy.

GIMM-VFI-F uses the CUDA/CuPy path where required by its implementation.

### 4.3 Provider CUDA semantics

RunPod/Novita provisioning should request a CUDA-13-capable host/driver where their API supports a CUDA compatibility field.

The provider-reported CUDA capability may be newer than the container userspace. That is acceptable if the NVIDIA driver can run the selected container CUDA runtime.

---

## 5. GPU architecture policy

Support these architecture families:

```text
Ampere    -> useful <=48 GB SKUs, primarily sm_86 class
Ada       -> sm_89
Blackwell -> sm_120 RTX/workstation class
```

Global hardware policy:

```text
VRAM <= 48 GB
```

Do not provision A100/H100/H200/B200-class workers or other >48 GB / high-cost datacenter SKUs outside this product policy.

The restriction is SKU/economics/VRAM, not a blanket architecture ban.

---

## 6. GPU-only execution invariant

Neural inference must never silently fall back to CPU.

Required runtime behavior:

```text
torch.cuda.is_available() == false
  -> worker unhealthy
  -> never advertise ready
  -> report error to control plane
  -> delete/replace pod

model parameters on CPU at inference boundary
  -> abort job

input tensors on CPU at inference boundary
  -> abort job

unexpected CPU/offload execution
  -> abort job
```

Do not use ComfyUI automatic model offloading semantics in the production worker.

The worker explicitly owns:

```text
CUDA_VISIBLE_DEVICES
cuda:0 selection
model.to(cuda:0)
input tensor placement
output tensor validation
```

CPU remains allowed for orchestration, HTTP, R2 I/O, JSON, filesystem work, some decode/encode/pre/post processing, and process supervision. Neural inference is GPU-only.

### 6.1 Startup smoke tests

Before `/ready` returns true, run real GPU checks.

FAST:

```text
CUDA allocation + CUDA op
CuPy CUDA kernel
Real-ESRGAN representative native GPU inference
RIFE native GPU inference and TensorRT engine check if present
GIMM-VFI-F native GPU inference
```

QUALITY:

```text
CUDA allocation + CUDA op
CuPy CUDA kernel
FlashVSR representative GPU inference
RIFE native GPU inference and TensorRT engine check if present
GIMM-VFI-F native GPU inference
```

Record:

```text
GPU name
VRAM
compute capability
driver version
CUDA runtime
PyTorch version
CuPy version
TensorRT version
model capability pass/fail
peak allocated VRAM
NVML telemetry
```

Do not trust a single sampled `nvidia-smi GPU-Util` number as proof of execution. Use actual CUDA tensor placement, CUDA events/synchronization, allocated process VRAM, and NVML telemetry.

---

## 7. TensorRT plan

### 7.1 Exact TensorRT version

**Pin TensorRT 10.16.1 for phase 1.**

Why:

- It is the newest TensorRT 10.x release.
- It retains the TensorRT 10.x FP16 / weak-typing builder workflow used by the RIFE and ESRGAN export patterns we intend to implement.
- It contains later CUDA-13 and Blackwell fixes than 10.14.1.
- TensorRT 11.x removes `BuilderFlag.FP16` and related weak-typing APIs and requires a strong-typing/ModelOpt migration.

References:

- https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.16.1.html
- https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-python-api-reference.html

Do not upgrade to TensorRT 11.x until our own exporters are intentionally migrated and parity-tested.

### 7.2 TensorRT `.engine` scope — exactly three model families

**Only these models may have TensorRT `.engine` files:**

```text
1. realesr-animevideov3
2. realesr-general-x4v3
3. RIFE
```

No other enhancer model gets an `.engine` file.

Explicitly prohibited TensorRT engine targets:

```text
RealESRGAN_x4plus_anime_6B image model -> NO .engine
RealESRGAN_x4plus image model          -> NO .engine
GIMM-VFI-F                             -> NO .engine
FlashVSR                               -> NO .engine
```

### 7.3 Image ESRGAN stays native PyTorch GPU

Storyboard image upscaling uses:

```text
RealESRGAN_x4plus_anime_6B
RealESRGAN_x4plus
```

These run native GPU PyTorch only. Do not export or cache TensorRT engines for them.

### 7.4 Video ESRGAN TensorRT

Director FAST video upscaling uses:

```text
realesr-animevideov3
realesr-general-x4v3
```

These are the only ESRGAN models that get ONNX/TensorRT export.

Requirements:

- export each video ESRGAN model ourselves;
- validate visual parity against native PyTorch;
- use FP16 TensorRT engines initially;
- build/cache engines per GPU architecture/profile;
- if the matching engine is unavailable or invalid, fall back only to native **GPU PyTorch**, never CPU.

### 7.5 RIFE TensorRT

RIFE is the third and only VFI model with TensorRT engines.

Initial target:

```text
RIFE implementation/checkpoint: pinned and parity-tested
precision: FP16
engine execution: TensorRT 10.16.1
```

Use permissively licensed upstream RIFE code to build our own export/runtime path. Community TensorRT wrappers may be used as implementation references/benchmarks only when their licenses do not permit direct commercial reuse.

Both FAST and QUALITY images contain the RIFE runtime and can consume the same architecture-specific engine cache format.

### 7.6 GIMM-VFI-F has no TensorRT engine

GIMM-VFI-F runs native CUDA PyTorch + CuPy only.

Do not export GIMM to ONNX/TensorRT in phase 1 and do not create `.engine` artifacts for it.

### 7.7 FlashVSR has no TensorRT engine

FlashVSR runs through its validated native CUDA / attention backend in QUALITY.

Do not create a TensorRT `.engine` for FlashVSR.

### 7.8 TensorRT engines are GPU-specific artifacts

Never assume one `.engine` file works across Ampere, Ada and Blackwell.

Engine cache key must include at least:

```text
model id
model/checkpoint hash
TensorRT version
CUDA major/minor
compute capability
precision
shape/profile id
```

Example cache namespace:

```text
enhancer-engines/{model}/{trtVersion}/{cuda}/{sm}/{precision}/{profile}.engine
```

Allowed `{model}` values are only:

```text
realesr-animevideov3
realesr-general-x4v3
rife
```

On pod startup:

1. Detect GPU and compute capability.
2. Look for a matching cached engine in R2.
3. Validate engine metadata.
4. If absent, build on that GPU or through an approved architecture-matched builder path.
5. Smoke-test the engine.
6. Upload the validated engine to the cache.
7. Advertise the TensorRT capability only after the test passes.

Do not bake a 4090 engine and expect it to run on a 5090.

---

## 8. Real-ESRGAN model map

There are exactly four ESRGAN models in the enhancer product.

### 8.1 Storyboard image models — native GPU PyTorch only

```text
Anime image -> RealESRGAN_x4plus_anime_6B
Real image  -> RealESRGAN_x4plus
```

No TensorRT engines for these two.

### 8.2 Director video models — TensorRT preferred, native GPU fallback

```text
Anime video -> realesr-animevideov3
Real video  -> realesr-general-x4v3
```

TensorRT engines are allowed only for these two ESRGAN video models.

### 8.3 BasicSR compatibility

Do not change ESRGAN architecture or weights.

The approved compatibility patch is the modern torchvision import relocation when required by the pinned BasicSR revision:

```python
# old
from torchvision.transforms.functional_tensor import rgb_to_grayscale

# new
from torchvision.transforms.functional import rgb_to_grayscale
```

This is dependency compatibility only.

After patching, the image build must run BasicSR import, Real-ESRGAN import, and representative CUDA inference tests.

---

## 9. RIFE implementation

RIFE is available in both FAST and QUALITY images.

Rules:

- GPU-only.
- TensorRT 10.16.1 FP16 engine path where a validated engine exists.
- Native CUDA PyTorch fallback only.
- Never CPU fallback.
- Pin the exact implementation/checkpoint hash.
- Startup capability test must prove real CUDA execution.

---

## 10. GIMM-VFI-F implementation

Selected variant:

```text
GIMM-VFI-F
flow estimator: FlowFormer
```

Do not silently substitute:

```text
GIMM-VFI-R
GIMM-VFI-F-P
GIMM-VFI-R-P
```

Run native PyTorch + CUDA-13 CuPy.

No TensorRT `.engine` files for GIMM.

### 10.1 Production license blocker

The original GIMM-VFI repository uses the S-Lab License 1.0 and is non-commercial by default unless commercial permission is obtained from the contributors.

Reference:
https://github.com/GSeanCDAT/GIMM-VFI/blob/main/LICENSE

Therefore:

```text
GIMM-VFI-F implementation/testing -> only as permitted by the license
commercial production enablement -> BLOCKED until commercial permission/license is confirmed
```

Do not hide or bypass this blocker.

---

## 11. FlashVSR implementation

FlashVSR exists only in QUALITY for video super-resolution.

Rules:

- native CUDA/PyTorch attention path;
- no TensorRT `.engine`;
- no CPU fallback;
- Ampere/Ada/Blackwell support only after a real startup smoke test passes on that architecture;
- keep FlashVSR-specific dependency and kernel compatibility inside the QUALITY image.

---

## 12. Video decode/encode

Bake FFmpeg with NVDEC/NVENC support where licensing/provider packaging allows it.

Preferred data path:

```text
NVDEC -> GPU inference -> NVENC
```

Avoid unnecessary host/device frame copies.

Software decode/encode fallback may be used for unsupported codecs/container work, but neural inference remains GPU-only.

---

## 13. Provider provisioning

Mirror the existing H3 operational pattern without sharing H3 state.

RunPod request characteristics:

```text
name
imageName
gpuTypeIds
gpuTypePriority=custom
gpuCount=1
computeType=GPU
allowed CUDA compatibility >= enhancer minimum
containerDiskInGb
env
ports
interruptible=false
locked=false
```

Novita request characteristics:

```text
name
productId
gpuNum
rootfsSize
imageUrl
ports
envs
minCudaVersion
billing settings
```

Provider env includes enhancer worker/control token information, service kind, R2 access, idle timeout, debug flags, and port configuration.

---

## 14. Lifecycle behavior

Enhancer lifecycle must include:

```text
priority dispatch
idle worker reuse
persistent admin idle timeout
provision timeout
job timeout
heartbeat
progress
debug/error state
retry policy
provider fallback
cancel
stale allocation recovery
idle reaper
provider delete lock
delete verification
delete retry/backoff
orphan cleanup
```

Timeout behavior must terminate/delete abandoned provider instances using the enhancer delete-lock table, not H3 locks.

---

## 15. Admin controls

Use the existing admin authorization policy (`ALLOWED_EMAILS`).

Expose enhancer admin controls for at least:

```text
idle timeout
jobs
priority
pods
stop/delete pod
manual dispatch/debug
runtime/provider/GPU capability data
TensorRT engine capability/version state
```

Idle timeout must be persistently configurable through enhancer config.

---

## 16. Storyboard contract

Storyboard image upscale writes back through existing fields only.

Rules:

- `scenes[n].image` becomes the active upscaled image.
- Preserve previous/original image state through existing original fields.
- Preserve storyboard thumbnail/original thumbnail behavior.
- Do not overload `isEnhanced` as image-upscale provenance.
- Do not add new enhancer-specific project JSON schema.

Upscale All uses enhancer pods only. Existing single-image `/api/upscale` remains the legacy Replicate path.

---

## 17. Director contract

Director video upscale writes only:

```text
project_video_timeline.segments[n].upscaledVideoUrl
```

Do not add enhancer metadata/provenance fields to Director JSON.

Automatic still-image rule:

```text
active media is video -> eligible
active media is image -> skipped_image
```

No pod/job/charge/video output is created for a skipped still image.

---

## 18. Build and deployment

Keep enhancer build/deploy isolated from the H3 runtime tree.

Enhancer Docker build context is the `enhancer/` directory only.

Build pipeline should produce:

```text
scenebuilder-enhancer-fast:latest
scenebuilder-enhancer-quality:latest
```

Do not make enhancer-only changes trigger H3 image rebuilds.

A Docker image merely building successfully is not enough. Promotion requires real GPU qualification.

Qualification matrix:

```text
Ampere
Ada
Blackwell
```

For each supported architecture, verify:

```text
CUDA
CuPy
PyTorch
GPU-only invariant
FAST model capabilities
QUALITY model capabilities
TensorRT 10.16.1 engine load/build for the allowed three model families
NVDEC/NVENC where available
```

---

## 19. Engine artifact policy

R2 may contain TensorRT engine artifacts only for:

```text
realesr-animevideov3
realesr-general-x4v3
rife
```

No `.engine` artifacts for any other enhancer model.

Engine artifacts are operational cache data, not product media state.

They must be keyed by architecture/runtime metadata and validated before use.

---

## 20. Source/license policy

Relevant upstream license review must be completed before commercial production use.

Current broad categories:

```text
RIFE         -> permissive upstream available; use upstream, not non-commercial wrapper code
Real-ESRGAN  -> BSD-3-Clause upstream
FlashVSR     -> verify/pin upstream license and dependency licenses
GIMM-VFI     -> S-Lab non-commercial by default; production blocked until permission/license is confirmed
```

Community ComfyUI/TensorRT wrappers are implementation references only unless their licenses explicitly permit commercial reuse.

---

## 21. Final canonical capability matrix

```text
FAST
  CUDA 13.x
  PyTorch CUDA-13
  CuPy cuda13x
  TensorRT 10.16.1

  IMAGE UPSCALE
    RealESRGAN_x4plus_anime_6B -> native GPU only
    RealESRGAN_x4plus          -> native GPU only

  VIDEO UPSCALE
    realesr-animevideov3       -> TensorRT .engine preferred; native GPU fallback
    realesr-general-x4v3       -> TensorRT .engine preferred; native GPU fallback

  VFI
    RIFE                       -> TensorRT .engine preferred; native GPU fallback
    GIMM-VFI-F                 -> native PyTorch + CuPy only

QUALITY
  CUDA 13.x
  PyTorch CUDA-13
  CuPy cuda13x
  TensorRT 10.16.1 for RIFE only

  VIDEO UPSCALE
    FlashVSR                   -> native GPU only

  VFI
    RIFE                       -> TensorRT .engine preferred; native GPU fallback
    GIMM-VFI-F                 -> native PyTorch + CuPy only
```

**No FILM.**

**TensorRT `.engine` files only for:**

```text
realesr-animevideov3
realesr-general-x4v3
RIFE
```

Everything else stays native CUDA GPU execution.

# SceneBuilder Enhancer Implementation Plan

Status: **canonical implementation plan**

Branch: `feat/enhancer-gpu-runtime`

This file is intentionally separate from the H3 runtime and H3 lifecycle design. Do not modify H3 lifecycle behavior to implement the enhancer.

---

## 1. Goal

Build a SceneBuilder enhancer runtime that behaves operationally like the existing H3 pod system, while remaining physically isolated from H3 state and preserving existing SceneBuilder D1/R2 product contracts.

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

Both images contain all supported VFI choices so a pod does not need to be replaced just because the user selects a different interpolation method.

### 3.1 FAST image

```text
CUDA 13.x userspace
PyTorch CUDA 13 build
CuPy CUDA 13 package
TensorRT 10.x compatibility runtime/builder
FFmpeg + NVDEC/NVENC

Real-ESRGAN image models:
  1. RealESRGAN_x4plus_anime_6B
  2. RealESRGAN_x4plus

Real-ESRGAN video models:
  3. realesr-animevideov3
  4. realesr-general-x4v3

VFI:
  5. RIFE
  6. FILM (PyTorch implementation)
  7. GIMM-VFI-F
```

### 3.2 QUALITY image

```text
CUDA 13.x userspace
PyTorch CUDA 13 build
CuPy CUDA 13 package
TensorRT runtime/builder where useful
FFmpeg + NVDEC/NVENC

Video upscale:
  1. FlashVSR

VFI:
  2. RIFE
  3. FILM (PyTorch implementation)
  4. GIMM-VFI-F
```

### 3.3 VFI product choices

The user-facing VFI choices are independent of FAST/QUALITY upscale mode:

```text
RIFE       -> fastest / TensorRT-optimized path where validated
FILM       -> native PyTorch GPU path
GIMM-VFI-F -> highest-quality/slower path; native PyTorch + CuPy
```

Use **GIMM-VFI-F**, not GIMM-VFI-R and not the perceptual `F-P` variant, unless product requirements are deliberately changed later.

The Reddit comparison that drove this choice reports the F model as slower but higher quality than R, FILM and RIFE, especially under larger motion:

https://www.reddit.com/r/StableDiffusion/comments/1j2evqn/wan_14b_with_mmaudio_gimmvfif_frame_interpolation/

---

## 4. CUDA baseline

### 4.1 Primary target

Use CUDA 13 as the enhancer baseline.

Target baseline for implementation:

```text
CUDA userspace: 13.0.x minimum
PyTorch: 2.12 stable CUDA-13 build or later validated compatible release
CuPy: cupy-cuda13x
NVIDIA driver requirement: CUDA-13-compatible provider driver
```

PyTorch 2.12 keeps CUDA 13.0 as the stable/default wheel and deprecated CUDA 12.8 in the standard matrix.

Reference:
https://pytorch.org/blog/pytorch-2-12-release-blog/

### 4.2 Never install two CUDA major lines in one image

Do not create this:

```text
CUDA 11 + CUDA 12.8 + CUDA 13
```

There is one system CUDA userspace line per Docker image.

Do not install both:

```text
cupy-cuda12x
cupy-cuda13x
```

Only install the CUDA-13 CuPy package.

Current CuPy provides `cupy-cuda13x` packages for CUDA 13.x.

Reference:
https://github.com/cupy/cupy

### 4.3 GIMM/CuPy compatibility

GIMM-VFI uses CuPy in the current ecosystem. The Kijai ComfyUI wrapper historically hard-pinned CUDA-12 CuPy, and there is a public 2026 compatibility report showing that installing both `cupy-cuda12x` and `cupy-cuda13x` breaks the environment on RTX 5090.

Reference:
https://github.com/kijai/ComfyUI-GIMM-VFI/pull/33

Our runtime must own CuPy installation directly and must not let a third-party installer downgrade or double-install it.

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

Do not provision the expensive datacenter classes that are outside this policy, including A100/H100/H200/B200-class workers.

The restriction is SKU/economics/VRAM, not a blanket architecture ban.

### 5.1 Provider CUDA requirement

RunPod/Novita provisioning should request a CUDA-13-capable host/driver where their API supports a CUDA compatibility field.

A provider host may advertise a newer CUDA-capable driver than the container userspace. That is acceptable if the NVIDIA driver is backward compatible with the CUDA runtime in the image.

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

CPU is still allowed for orchestration, HTTP, R2 I/O, JSON, filesystem work, some decode/encode/pre/post processing, and process supervision. The neural model execution itself is GPU-only.

### 6.1 Startup smoke tests

Before `/ready` returns true, run real GPU checks:

FAST:

```text
CUDA allocation + CUDA op
CuPy CUDA kernel
Real-ESRGAN representative inference
RIFE inference
FILM inference
GIMM-VFI-F inference
```

QUALITY:

```text
CUDA allocation + CUDA op
CuPy CUDA kernel
FlashVSR representative inference
RIFE inference
FILM inference
GIMM-VFI-F inference
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
```

Do not trust a single sampled `nvidia-smi GPU-Util` number as proof of execution. Use actual CUDA tensor placement, CUDA events/synchronization, allocated process VRAM, and NVML telemetry.

---

## 7. TensorRT plan

### 7.1 Which TensorRT version

**Phase 1 target: TensorRT 10.14.1.48 on CUDA 13.x.**

Reason:

- TensorRT 10.14.1 is proven on CUDA 13.0/13.1-era stacks.
- The current RIFE TensorRT and ESRGAN-class community implementations use the TensorRT 10.x weak-typing/FP16 builder style.
- TensorRT 11 removes `BuilderFlag.FP16` and related weak-typing APIs, so blindly jumping to TensorRT 11 breaks those exporters/builders.

References:

https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.14.1.html
https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-python-api-reference.html

TensorRT 11.x is a later migration task after we own strongly-typed ONNX/ModelOpt export paths.

### 7.2 TensorRT targets

Use TensorRT where it has a mature and measurable benefit:

```text
RIFE             -> YES, primary TensorRT acceleration target
Real-ESRGAN x4   -> YES, validate/export each of the four selected models
FILM             -> native PyTorch first
GIMM-VFI-F       -> native PyTorch + CuPy, no TensorRT initially
FlashVSR         -> native PyTorch/block-sparse path, no TensorRT initially
```

### 7.3 RIFE TensorRT

The current public RIFE TensorRT project is tested on CUDA 13.0/PyTorch 2.12 and supports `rife49`, `rife48`, and `rife47`, with `rife49_ensemble_True_scale_1_sim` as its accuracy-oriented default.

Reference:
https://github.com/yuvraj108c/ComfyUI-Rife-Tensorrt

For SceneBuilder, the first TensorRT speed target is:

```text
rife49_ensemble_True_scale_1_sim
precision: FP16
```

Keep the product API named `RIFE`; implementation can later move to a newer RIFE export after parity testing.

Do not vendor the third-party ComfyUI TensorRT node code directly into a commercial product because that project is CC BY-NC-SA. Use it as a benchmark/reference and build our own exporter/runtime from the permissively licensed RIFE source, or obtain appropriate permission.

### 7.4 Real-ESRGAN TensorRT

TensorRT acceleration should be validated separately for all four selected Real-ESRGAN models:

```text
image anime: RealESRGAN_x4plus_anime_6B
image real:  RealESRGAN_x4plus
video anime: realesr-animevideov3
video real:  realesr-general-x4v3
```

A public TensorRT upscaler implementation has demonstrated ESRGAN-architecture TensorRT on RTX 5090/CUDA 13.1 with 2-4x class speedups, but its tested model list is not identical to our four models.

Reference:
https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt

Therefore:

- export each selected model ourselves to ONNX;
- validate numerical/visual parity;
- build FP16 TensorRT engines per GPU architecture;
- fall back only to native **GPU PyTorch**, never CPU, if a specific ESRGAN model cannot yet use TensorRT.

### 7.5 Why GIMM-VFI-F is not TensorRT in phase 1

There is no mature GIMM-VFI TensorRT path today. A public request for GIMM-VFI TensorRT was closed as not planned.

Reference:
https://github.com/yuvraj108c/ComfyUI-Rife-Tensorrt/issues/10

GIMM-VFI-F also relies on a more complex FlowFormer/CuPy path. Do not force it through TensorRT just to make the stack uniform.

Run it as native CUDA PyTorch + CuPy and optimize later only if a reliable exporter exists.

### 7.6 TensorRT engines are GPU-specific artifacts

Never assume one `.engine` file works across Ampere, Ada and Blackwell.

TensorRT can reject an engine built for one compute capability when deserialized on another GPU.

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

On pod startup:

1. Detect GPU and compute capability.
2. Look for a matching cached engine in R2.
3. Validate engine metadata.
4. If absent, build on that GPU.
5. Smoke-test the engine.
6. Upload the validated engine to the cache.
7. Advertise the capability only after the test passes.

Do not bake a 4090 engine into an image and expect it to run on a 5090.

---

## 8. Real-ESRGAN / BasicSR compatibility

Do not change ESRGAN model architecture or weights.

The only approved BasicSR compatibility patch is the modern torchvision import relocation required by current torchvision:

```python
# old
from torchvision.transforms.functional_tensor import rgb_to_grayscale

# new
from torchvision.transforms.functional import rgb_to_grayscale
```

This is a dependency compatibility patch only.

After patching, the Docker build must run:

```text
BasicSR import test
Real-ESRGAN import test
representative CUDA inference test
```

---

## 9. FILM implementation

Do not ship Google's old TensorFlow 2.6 / CUDA-11-era runtime.

Use a PyTorch FILM implementation/port and keep FILM inside the same CUDA-13 PyTorch environment as the other enhancer models.

FILM must run on GPU only and must pass the same architecture-specific smoke tests.

TensorRT for FILM is not a phase-1 requirement.

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

Use native PyTorch + CUDA-13 CuPy.

GIMM-F is slower than RIFE/FILM and is the quality-oriented VFI choice.

### 10.1 Important production-license blocker

The original GIMM-VFI repository uses the **S-Lab License 1.0**, which allows redistribution/use for non-commercial purposes and requires contacting the contributors for commercial use.

Reference:
https://github.com/GSeanCDAT/GIMM-VFI/blob/main/LICENSE

Therefore:

```text
GIMM-VFI-F implementation/testing: allowed in development subject to license terms
commercial production enablement: BLOCKED until commercial permission/license is confirmed
```

Do not hide this blocker. A wrapper repository does not remove the original model/code license obligation.

---

## 11. Licensing/source policy

Current relevant upstreams:

```text
RIFE         -> MIT upstream
Real-ESRGAN  -> BSD-3-Clause upstream
FILM         -> Apache-2.0 upstream
FlashVSR     -> Apache-2.0 upstream
GIMM-VFI     -> S-Lab non-commercial by default
```

Community TensorRT ComfyUI wrappers used for benchmarking may have non-commercial licenses. Do not copy them into the SceneBuilder commercial runtime unless their license permits it or permission is obtained.

Use permissively licensed upstream models and write our own thin inference/export adapters where necessary.

---

## 12. Video pipeline

Use GPU video paths where practical:

```text
FFmpeg demux
NVDEC decode where supported
CUDA inference
NVENC encode where supported
FFmpeg mux
```

Software decode/encode fallback is allowed when a codec/container is unsupported by NVDEC/NVENC.

That fallback does not permit CPU neural inference.

For interpolation pipelines, keep intermediate frames on GPU where practical to reduce PCIe copies.

---

## 13. Provider provisioning

Follow the H3 provider model while keeping enhancer state separate.

### RunPod request shape

Keep the same operational pattern as H3:

```text
name
imageName
gpuTypeIds
gpuTypePriority
gpuCount=1
computeType=GPU
cloud type fallback
allowedCudaVersions / CUDA compatibility field
container disk
env
ports
interruptible=false
locked=false
```

### Novita request shape

Keep the same operational pattern as H3:

```text
name
productId
gpuNum=1
rootfsSize
imageUrl
ports
envs
command/tools
cluster/network fields
billing fields
minCudaVersion / CUDA compatibility field
```

Enhancer runtime env includes:

```text
R2 credentials
worker id
enhancer pod token
enhancer control-plane URL
service kind
idle timeout
port
debug flags
GPU-only enforcement flags
```

---

## 14. Enhancer pod lifecycle

Required lifecycle states/behavior:

- provision worker row before provider request finishes;
- provider allocation timeout;
- provider fallback RunPod <-> Novita;
- worker readiness/health validation;
- heartbeat;
- priority queue dispatch;
- one active GPU job per worker;
- idle worker reuse;
- configurable idle timeout;
- job timeout;
- cancel;
- retry/transient failure handling;
- permanent failure handling;
- debug/error fields;
- provider deletion lock;
- deletion verification;
- exponential/backoff delete retries;
- stale worker recovery;
- orphan provider instance cleanup.

Separate callback endpoint:

```text
/api/projects/v2/enhancer/pod/events
```

Callback events include:

```text
worker_ready
heartbeat
job_progress
job_completed
job_failed
worker_idle
idle_expired
error
```

Use timestamp + nonce replay protection.

---

## 15. Admin requirements

Admin must expose enhancer controls without mixing them into H3 rows.

Required controls/views:

```text
idle timeout
pods list
jobs list
job priority
manual dispatch
stop/delete pod
provider
GPU model
VRAM
compute capability
CUDA runtime
PyTorch/CuPy/TensorRT versions
capabilities
current job
heartbeat
progress
errors
debug logs / debug summary
delete retry state
```

Use the existing `ALLOWED_EMAILS` admin authorization behavior.

Persistent enhancer config key:

```text
idle_timeout_seconds
```

---

## 16. Storyboard image writeback

Use existing fields only.

For an upscaled scene:

```text
scenes[n].image = new active upscaled image
preserve previous/base image as originalImage
storyboard.thumbnail may be updated
preserve storyboard.originalImage
preserve storyboard.originalThumbnail
```

Do not overload `isEnhanced` to mean image-upscale provenance.

Do not introduce enhancer-specific product JSON fields.

---

## 17. Director video writeback

Use only the existing field:

```text
project_video_timeline.segments[n].upscaledVideoUrl
```

Do not add enhancement metadata/provenance fields to the Director JSON contract.

Automatic still-image skip:

```text
active/current media is video -> eligible
active/current media is image -> skipped_image
```

Skipped images must not create a GPU job, pod charge, output video, or writeback.

---

## 18. Queue/job model

Local enhancer jobs use `pending_upscales` with an enhancer-specific discriminator/state.

New local jobs begin in a state that legacy Replicate cron code cannot claim, e.g.:

```text
waiting_for_pod
```

Priority ordering:

```text
priority DESC
created_at ASC
```

Job kinds include:

```text
image_upscale
video_upscale
video_vfi
benchmark / capability test
```

Operational fields include provider, worker, GPU, runtime, engine/model, progress, timeout, retries, cancellation, errors and attempt history.

---

## 19. Image and video model routing

### Storyboard image upscale

```text
anime -> RealESRGAN_x4plus_anime_6B
real  -> RealESRGAN_x4plus
```

### Director FAST video upscale

```text
anime -> realesr-animevideov3
real  -> realesr-general-x4v3
```

### Director QUALITY video upscale

```text
FlashVSR
```

### Optional VFI after video upscale

```text
RIFE
FILM
GIMM-VFI-F
```

The same VFI selector must work whether the video upscale mode is FAST or QUALITY.

---

## 20. Capability advertisement

A worker advertises only capabilities that actually passed startup/runtime validation.

Example FAST Blackwell capability payload:

```json
{
  "gpuOnly": true,
  "architecture": "blackwell",
  "computeCapability": "12.0",
  "cuda": "13.x",
  "cupy": true,
  "tensorrt": true,
  "esrganImageAnime": true,
  "esrganImageReal": true,
  "esrganVideoAnime": true,
  "esrganVideoReal": true,
  "rife": true,
  "film": true,
  "gimmVfiF": true,
  "nvdec": true,
  "nvenc": true
}
```

Example QUALITY capability payload adds:

```text
flashvsr
```

Do not advertise a capability just because its package imported successfully. It must pass actual GPU inference.

---

## 21. Build and release strategy

Enhancer code stays under:

```text
enhancer/
```

Do not alter the H3 Dockerfile tree to make enhancer builds work.

The enhancer workflow builds FAST and QUALITY independently and triggers only for enhancer paths/workflow changes.

Required validation layers:

```text
1. source/unit tests
2. Docker build
3. Python import tests
4. dependency/version assertions
5. checkpoint hash verification
6. GPU integration qualification on real hardware
7. only then publish/mark image scheduler-eligible
```

A Docker build succeeding is not enough.

### 21.1 Real GPU qualification matrix

At minimum qualify:

```text
Ampere <=48 GB
Ada <=48 GB
Blackwell <=48 GB
```

For each architecture verify:

FAST:

```text
4x ESRGAN
RIFE native
RIFE TensorRT
FILM
GIMM-VFI-F
CuPy kernel
NVDEC/NVENC
```

QUALITY:

```text
FlashVSR
RIFE native
RIFE TensorRT
FILM
GIMM-VFI-F
CuPy kernel
NVDEC/NVENC
```

TensorRT engine artifacts are built/cached per architecture, not copied blindly across GPUs.

---

## 22. Failure policy

Examples:

```text
CUDA unavailable
  -> worker unhealthy -> delete/reprovision

CuPy CUDA test fails
  -> dependent capability false; if required capability, worker unhealthy

model runs on CPU
  -> hard job failure; never continue on CPU

TensorRT engine incompatible with GPU
  -> discard engine; rebuild for exact GPU; native GPU fallback only if allowed

provider provisioning timeout
  -> delete provider instance -> retry/fallback provider

job timeout
  -> cancel runtime -> mark/retry according to policy -> delete/reuse worker safely

permanent model/runtime error
  -> no pointless retry loop
```

Errors and debug information must be persisted for admin inspection.

---

## 23. Known compatibility risks to test before merge

- Real-ESRGAN/BasicSR import compatibility with current torchvision.
- Exact ONNX/TensorRT export compatibility of all four selected Real-ESRGAN models.
- RIFE TensorRT version compatibility; public RIFE TRT code does not currently cover every newer RIFE checkpoint.
- TensorRT 11 migration is intentionally deferred because 11.x removes old FP16 builder flags.
- GIMM-VFI-F CUDA-13 CuPy compatibility must be verified without installing `cupy-cuda12x`.
- GIMM-VFI-F commercial license must be resolved before production enablement.
- FlashVSR sparse-attention build on Blackwell `sm_120` must pass real inference, not only compile.
- Engine cache must never reuse an engine across incompatible compute capabilities.
- No startup `worker_idle` event may clear an assigned provisioning job before `worker_ready`/submission.
- Pod callback ordering must not allow worker reuse before GPU work is actually finished.

---

## 24. Control-plane implementation order

1. Finish enhancer D1 schema and migration.
2. Finish enhancer worker/job store.
3. Finish provider routing with <=48 GB GPU policy and CUDA-13 requirement.
4. Finish callback auth/replay protection.
5. Finish priority dispatch, retries, timeouts, delete locks and idle lifecycle.
6. Finish Storyboard Upscale All local routing with zero Replicate.
7. Finish Director video routing/writeback and still-image skip.
8. Finish enhancer admin API and visible admin idle-timeout control.
9. Finish runtime capability protocol.
10. Finish FAST image with 4x ESRGAN + RIFE + FILM + GIMM-VFI-F.
11. Add RIFE TensorRT engine builder/cache.
12. Add/validate Real-ESRGAN TensorRT exports.
13. Finish QUALITY image with FlashVSR + RIFE + FILM + GIMM-VFI-F.
14. Add GPU qualification jobs for Ampere/Ada/Blackwell.
15. Resolve GIMM-VFI commercial license before enabling it in production.
16. Merge runtime only after GPU image qualification succeeds.
17. Merge SceneBuilder control plane only after required runtime images are actually available.
18. Monitor Cloudflare/CI deployment and provider startup after merge.

---

## 25. Definition of done

The enhancer is done only when all of the following are true:

- H3 lifecycle files remain behaviorally unchanged.
- Enhancer lifecycle uses enhancer-only D1 tables.
- Upscale All never calls Replicate.
- Single-image legacy upscale still works through its existing path.
- FAST has four selected ESRGAN models plus RIFE, FILM and GIMM-VFI-F.
- QUALITY has FlashVSR plus RIFE, FILM and GIMM-VFI-F.
- GIMM variant is exactly `GIMM-VFI-F` unless deliberately changed.
- CUDA 13 is the runtime baseline.
- CuPy uses one CUDA-13 package only.
- No neural model silently runs on CPU.
- TensorRT is used for validated RIFE/ESRGAN paths and never reused across incompatible GPUs.
- Ampere/Ada/Blackwell <=48 GB qualification exists.
- Priority, idle timeout, timeout deletion, retries, provider fallback, debug/error state and admin control behave H3-style.
- Existing D1/R2 product contracts are preserved.
- GIMM-VFI-F remains production-disabled until its commercial license is cleared.
- CI/GPU smoke tests pass before main/deployment.

---

## 26. Research/reference links

GIMM-VFI original:
https://github.com/GSeanCDAT/GIMM-VFI

GIMM-VFI-F Reddit example/comparison:
https://www.reddit.com/r/StableDiffusion/comments/1j2evqn/wan_14b_with_mmaudio_gimmvfif_frame_interpolation/

Kijai GIMM-VFI wrapper / CuPy CUDA-13 conflict:
https://github.com/kijai/ComfyUI-GIMM-VFI
https://github.com/kijai/ComfyUI-GIMM-VFI/pull/33

RIFE TensorRT reference/benchmark:
https://github.com/yuvraj108c/ComfyUI-Rife-Tensorrt

Real-ESRGAN TensorRT-class reference/benchmark:
https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt

PyTorch CUDA-13 baseline:
https://pytorch.org/blog/pytorch-2-12-release-blog/

CuPy CUDA-13 package:
https://github.com/cupy/cupy

TensorRT 10.14.1:
https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.14.1.html

TensorRT 10 -> 11 migration:
https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-python-api-reference.html

---

## 27. Plan change rule

When a future chat or implementation decision changes the enhancer architecture, update **this file** (`enhancer/PLAN.md`) in the H3 runtime repository first.

Do not put enhancer design changes into an H3 plan/lifecycle document.

# SceneBuilder Enhancer Implementation Plan

Status: **canonical implementation plan**

Branch: `feat/enhancer-gpu-runtime`

This file is intentionally separate from the H3 runtime/lifecycle design. Enhancer code lives in the same GPU-runtime repository, but enhancer lifecycle state and H3 lifecycle state remain physically separate.

This revision incorporates the approved operational/UI/TensorRT details from the older enhancer plans while keeping the newer locked decisions: **no FILM, no TensorRT for Storyboard image ESRGAN, RIFE 4.9, GIMM-VFI-F, CUDA 13.x, TensorRT 10.16.1, and GPU-only neural inference.**

---

## 1. Product goal

Build a SceneBuilder enhancer backend that behaves operationally like the existing H3 pod system without putting enhancer jobs/workers into H3 state.

Supported product paths:

- Storyboard still-image upscale.
- Video Generation Timeline / Director video upscale.
- Video frame interpolation.
- FAST and QUALITY local GPU runtimes.
- Existing Topaz external/premium upscale/VFI choices remain available through their existing external path.
- RunPod and Novita local enhancer pod provisioning.
- H3-style priority, retries, debug/error state, heartbeat, idle reuse, configurable idle timeout, job timeout, provider fallback, delete locks, delete verification/backoff, and orphan cleanup.

Neural inference on a paid enhancer GPU pod must never silently run on CPU.

---

## 2. Hard non-negotiables

### 2.1 H3 lifecycle remains untouched

H3 state remains:

```text
h3_pod_workers
video_generation_batches
video_generation_jobs
```

Enhancer state is separate:

```text
enhancer_pod_workers
pending_upscales
enhancer_pod_delete_locks
enhancer_event_nonces
enhancer_config
```

Rules:

- Never insert enhancer workers into `h3_pod_workers`.
- H3 code never queries `enhancer_pod_workers`.
- Enhancer allocation never treats `h3_pod_workers` as its worker pool.
- Enhancer has its own callback/replay/delete-lock/reaper state.
- Enhancer may reuse stateless provider/R2 helper patterns from H3, but not H3 stateful lifecycle rows/functions.
- Do not modify mature H3 lifecycle behavior to make enhancer work.

### 2.2 Existing product D1/JSON contracts remain authoritative

Do not add enhancer-specific product schema to:

```text
projects_timeline
project_video_timeline
```

Do not create a second project-media state table. Enhancer-specific D1 changes are operational only.

### 2.3 Existing product R2 layout remains authoritative

Permanent product output locations remain resolved through current media-layout rules. Canonical V2 destinations are:

```text
Storyboard image upscale:
projects/{projectId}/images/scenes/upscaled/

Director video upscale:
projects/{projectId}/video/upscaled/
```

No R2 product-media migration is part of enhancer work.

### 2.4 Replicate routing

- Storyboard **Upscale All** uses local enhancer pods only.
- No Replicate worker/cron may claim enhancer batch jobs.
- Existing single-image `/api/upscale` remains the legacy Replicate path unless deliberately changed later.

### 2.5 FILM is deleted

FILM is not installed, exposed, advertised, benchmarked, routed, or stored anywhere in the enhancer runtime/product.

---

## 3. Final model/runtime topology

There are two local enhancer Docker image families:

```text
scenebuilder-enhancer-fast
scenebuilder-enhancer-quality
```

### 3.1 FAST image

```text
CUDA 13.x userspace
PyTorch 2.x CUDA-13 build selected/validated for the final image
CuPy cupy-cuda13x
TensorRT 10.16.1 runtime + builder
FFmpeg with NVDEC/NVENC where available
NVML telemetry

Storyboard image upscale:
  RealESRGAN_x4plus_anime_6B
  RealESRGAN_x4plus

Director video upscale:
  realesr-animevideov3
  realesr-general-x4v3

VFI:
  RIFE 4.9
  GIMM-VFI-F
```

### 3.2 QUALITY image

```text
CUDA 13.x userspace
PyTorch 2.x CUDA-13 build selected/validated for the final image
CuPy cupy-cuda13x
TensorRT 10.16.1 runtime for RIFE
FFmpeg with NVDEC/NVENC where available
NVML telemetry

Video upscale:
  FlashVSR

VFI:
  RIFE 4.9
  GIMM-VFI-F
```

### 3.3 Topaz remains external

Keep existing premium/external choices behind their existing SceneBuilder path:

```text
Topaz Gaia 2
Topaz Proteus
Topaz Chronos where supported
Topaz Apollo where supported
```

Do not bake or redistribute Topaz assets in local enhancer Docker images.

---

## 4. CUDA / PyTorch / CuPy / TensorRT baseline

### 4.1 One CUDA line per Docker image

Use one CUDA userspace line in each enhancer Docker image.

```text
CUDA userspace: 13.x
PyTorch: 2.x build for the selected CUDA-13 runtime
CuPy: cupy-cuda13x
TensorRT: 10.16.1
```

Do not install CUDA 11/12/13 side by side in one image. `PyTorch 2.x` is the framework version; its CUDA build/runtime tag is a separate compatibility dimension.

### 4.2 CuPy

Install only `cupy-cuda13x`. Do not allow third-party installers to add `cupy-cuda12x` or another CuPy CUDA family beside it. GIMM-VFI-F uses the validated PyTorch/CuPy CUDA path.

### 4.3 TensorRT version

Pin **TensorRT 10.16.1**. Do not move to TensorRT 11.x until our exporters/runtime are intentionally migrated and parity-tested.

### 4.4 GPU architecture / VRAM policy

```text
Ampere    -> primarily sm_86 allowed <=48 GB SKUs
Ada       -> sm_89 allowed <=48 GB SKUs
Blackwell -> sm_120 RTX/workstation allowed <=48 GB SKUs
```

**48 GB GPUs are allowed.** Do not provision excluded high-cost datacenter classes such as A100/H100/H200/B200 or >48 GB workers for this product.

### 4.5 FlashVSR QUALITY floor

FlashVSR production routing starts at the approved RTX-4090-class floor or a validated equal/higher allowed GPU. Do not route FlashVSR to lower FAST-pool cards merely because they have enough nominal VRAM.

---

## 5. GPU-only execution invariant

Neural inference must never silently fall back to CPU.

```text
torch.cuda.is_available() == false
  -> worker unhealthy
  -> do not advertise ready
  -> report/requeue/fail according to policy
  -> terminate/replace pod

model parameters on CPU at inference boundary -> abort job
input tensors on CPU at inference boundary    -> abort job
unexpected framework CPU/offload path         -> abort job
```

Explicitly own `CUDA_VISIBLE_DEVICES`, `cuda:0`, model device, input tensor device, output validation, and CUDA synchronization/events.

CPU remains valid for HTTP/orchestration, R2 I/O, JSON/filesystem, probing, audio, and codec/pre/post work. It is not a neural fallback.

### 5.1 Startup qualification

Before `/ready` becomes true, run actual GPU tests.

FAST:

```text
CUDA allocation + CUDA op
CuPy CUDA kernel
Storyboard ESRGAN native CUDA inference
video ESRGAN native CUDA inference
RIFE 4.9 native CUDA inference
RIFE TensorRT deserialize/inference when a compatible engine is present
GIMM-VFI-F native CUDA/CuPy inference
NVENC smoke test when configured
```

QUALITY:

```text
CUDA allocation + CUDA op
CuPy CUDA kernel
FlashVSR representative inference
RIFE 4.9 native CUDA inference
RIFE TensorRT deserialize/inference when a compatible engine is present
GIMM-VFI-F native CUDA/CuPy inference
NVENC smoke test when configured
```

Advertise only capabilities that actually pass.

---

## 6. Exact TensorRT scope

TensorRT `.engine` files are allowed for exactly three model families:

```text
1. realesr-animevideov3
2. realesr-general-x4v3
3. RIFE 4.9
```

No `.engine` for:

```text
RealESRGAN_x4plus_anime_6B  # Storyboard image
RealESRGAN_x4plus           # Storyboard image
GIMM-VFI-F
FlashVSR
```

### 6.1 Storyboard image ESRGAN is native PyTorch CUDA only

```text
Anime -> RealESRGAN_x4plus_anime_6B
Real  -> RealESRGAN_x4plus
```

No ONNX/TensorRT engine generation, R2 engine folder, or TensorRT admin build option for these image models.

### 6.2 Video ESRGAN uses full-frame static engines

```text
Anime -> realesr-animevideov3
Real  -> realesr-general-x4v3
```

Primary strategy:

- FP16 TensorRT.
- Full-frame static video engines.
- No tiled TensorRT path.
- No separate engine for duration/FPS/concurrency.
- Native GPU PyTorch fallback for non-matching/invalid/missing engines.
- Never CPU fallback.

Do not use tiled ESRGAN/RIFE inference in the locked video paths. If full-frame execution cannot fit after concurrency backoff, requeue to a larger allowed GPU or fail/retry according to policy.

### 6.3 `general-x4v3` denoise policy

Use one fixed balanced denoise network/preset for the V1 TensorRT export. Do **not** create extra `.engine` variants for arbitrary DNI/denoise values.

### 6.4 BasicSR compatibility patch

Do not alter Real-ESRGAN architecture, weights, or quality behavior to make the modern runtime work. The approved BasicSR compatibility patch, when required by the pinned revision, is only:

```python
# old
from torchvision.transforms.functional_tensor import rgb_to_grayscale

# compatible
from torchvision.transforms.functional import rgb_to_grayscale
```

Pin official BasicSR/Real-ESRGAN source revisions and apply this deterministic compatibility patch during build. Validate BasicSR import, Real-ESRGAN import, and representative CUDA inference.

---

## 7. TensorRT engine topology

### 7.1 Video ESRGAN: 12 engines per compatibility target

For `realesr-animevideov3`:

```text
Landscape: 854x480, 1056x594, 1280x720
Portrait:  480x854, 594x1056, 720x1280
= 6 engines
```

For `realesr-general-x4v3`: same six shapes = 6 engines.

```text
6 anime + 6 real = 12 video ESRGAN engines
```

### 7.2 RIFE 4.9: 3 engines, not 6

RIFE uses three resolution-class engine files per compatibility target:

```text
1080 class
  landscape profile: 1920x1080
  portrait profile:  1080x1920

1440 / 2K class
  landscape profile: 2560x1440
  portrait profile:  1440x2560

2160 / 4K class
  landscape profile: 3840x2160
  portrait profile:  2160x3840
```

Each engine contains validated landscape/portrait profiles. Portrait support therefore does **not** double the file count.

```text
RIFE engine files = 3
```

Split into 6 only if later benchmarks prove separate orientation engines materially better; do not do so by default.

### 7.3 Total

Per TensorRT compatibility target:

```text
video ESRGAN anime = 6
video ESRGAN real  = 6
RIFE 4.9           = 3
----------------------
total              = 15 .engine files
```

If three separate same-compute-capability sets are generated for `sm_86`, `sm_89`, and `sm_120`, that is 45 physical engine files.

### 7.4 What does NOT create another engine

No additional engine for:

```text
video duration
number of frames
24 -> 48 FPS
24 -> 60 FPS
30 -> 48 FPS
30 -> 60 FPS
x2/x3/x4 frame concurrency
Director trim range
Director playback speed
```

Those are runtime scheduling/timestamp concerns.

---

## 8. TensorRT portability and lookup order

Support these compatibility targets where TensorRT 10.16.1 validates them:

```text
AMPERE_PLUS portable
SAME_COMPUTE_CAPABILITY / sm_86
SAME_COMPUTE_CAPABILITY / sm_89
SAME_COMPUTE_CAPABILITY / sm_120
Exact GPU
```

The allocator still filters through the SceneBuilder <=48 GB GPU allowlist. `AMPERE_PLUS` is an engine compatibility mode, not permission to provision excluded GPUs.

Runtime preference:

```text
1. exact-GPU validated active engine
2. same-compute-capability validated active engine
3. AMPERE_PLUS validated active engine
4. native PyTorch CUDA FP16 fallback
```

Never use an engine whose metadata/compatibility does not match the actual runtime/GPU.

---

## 9. Non-canonical video fallback

The 12 video ESRGAN engines are static shapes. A source that does not match an approved static shape must not be destructively resized merely to force a TensorRT hit unless a separate normalization rule has been explicitly approved for that source class.

```text
exact supported static shape -> TensorRT
no exact safe approved shape -> native PyTorch CUDA FP16 at source raster
```

Do not crush a 1920x1080 source to 1280x720 solely because only a 720 engine exists.

---

## 10. Output resolution and aspect handling

Neural scale and final product delivery resolution are separate.

Video targets:

```text
1080p
1440p / 2K
2160p / 4K
```

For video Real-ESRGAN, run the native x4 model and then perform final high-quality resize/conform to the exact requested output raster. Storyboard keeps existing 2K/4K product semantics.

Support both 16:9 and 9:16. Model identity is not tied to orientation. `ffprobe` actual video metadata instead of trusting requested/provider metadata blindly.

---

## 11. RIFE 4.9 timing behavior

Pin the exact RIFE 4.9 implementation revision and checkpoint SHA-256 used by native PyTorch and TensorRT export. Do not opportunistically download a different/latest checkpoint at pod startup.

RIFE must support arbitrary interpolation timestamps, not only integer `--multi`.

For source FPS `S`, target FPS `T`, and playback speed `p`:

```text
sourcePositionStep = S * p / T
sourcePosition = outputTime * S * p
left  = floor(sourcePosition)
right = left + 1
t     = sourcePosition - left
```

Invoke the same RIFE model/engine with the required fractional timestep `t`.

The same engine supports 24→48, 24→60, 30→48, 30→60, and other supported mappings.

### 11.1 Director range/playback-speed semantics

- Full-master enhancement preserves the authoritative full-master duration.
- If the Director action requests the current Director segment/range, process that authoritative trim/range rather than assuming the whole source.
- If `Apply Director playback speed` is enabled, VFI scheduling incorporates that playback speed.
- If disabled, VFI runs at normal source/master timing and Director speed remains a later timeline operation.
- No extra RIFE engine is created for trim, duration, or speed.

Output duration/frame count follows the selected full-master or Director timing policy exactly.

---

## 12. GIMM-VFI-F

Selected variant:

```text
GIMM-VFI-F
FlowFormer estimator
native PyTorch + CuPy CUDA
```

Do not substitute GIMM-VFI-R or perceptual `*-P` variants. No TensorRT engine for GIMM.

The original GIMM-VFI license is non-commercial by default unless required commercial permission is obtained; commercial production enablement remains blocked until that is cleared.

---

## 13. FlashVSR

FlashVSR exists only in QUALITY.

```text
native CUDA/PyTorch/custom-attention path
NO TensorRT .engine
NO CPU neural fallback
QUALITY GPU floor starts at approved 4090-class or higher
GPU/architecture capability only after real self-test
```

Do not reintroduce old FlashVSR TensorRT ideas in V1.

---

## 14. Model artifacts, R2 and D1 ownership

### 14.1 Docker contains durable source/model artifacts

FAST contains the four ESRGAN weight sets, RIFE 4.9 code/weights plus trusted ONNX/export source, GIMM-VFI-F permitted assets, video-ESRGAN ONNX/export artifacts, and TensorRT runtime/builder.

QUALITY contains FlashVSR code/weights/custom runtime, RIFE 4.9 code/weights plus trusted ONNX/export source, GIMM-VFI-F permitted assets, and TensorRT runtime for RIFE.

Do not bake generated `.engine` binaries into Docker.

### 14.2 Private engine R2 namespace

For model artifacts, R2 stores generated `.engine` binaries, TensorRT timing caches where useful, and optional immutable diagnostic sidecars. Model weights stay baked in Docker.

Product media continues using the existing SceneBuilder project R2 layout from Section 2.3.

### 14.3 No provider network volume

Do not require RunPod or Novita network volumes.

```text
Docker = durable model/source artifacts
R2     = generated engine/timing-cache artifacts + existing product media
pod disk = temporary working/download cache only
```

The temporary local cache directory is an implementation detail, not a product contract.

---

## 15. Private R2 engine layout

Use:

```text
models/.engine/
├── esrgananime/
├── esrganreal/
└── rife-4.9/
```

Do **not** create image-ESRGAN engine directories.

Compatibility directories:

```text
ampere-plus/
ampere-cc86/
ada-cc89/
blackwell-cc120/
exact/<gpu-slug>/
```

Use immutable filenames containing model/hash/TRT/CUDA/precision/profile/build identity, for example:

```text
v3__onnx-<sha>__trt-10.16.1__cuda-13__fp16__854x480__e-<id>.engine
rife49__onnx-<sha>__trt-10.16.1__cuda-13__fp16__1080-class__e-<id>.engine
```

Engine identity includes model/version, checkpoint/ONNX hash, TensorRT version, CUDA compatibility, precision, hardware compatibility mode, compute capability/exact GPU when relevant, shape/profile, and builder/plugin config hash.

---

## 16. D1 operational schema

### 16.1 Enhancer workers

Maintain separate `enhancer_pod_workers` with enhancer-only provider lifecycle data including service kind, provider/instance ID, endpoint, GPU/compute capability, runtime image/digest, CUDA, status/current job, capabilities/telemetry, heartbeat/error/debug, idle timeout, terminate/delete retry state, and timestamps.

### 16.2 Unified enhancer jobs stay in `pending_upscales`

Use the existing physical table `pending_upscales`. Do not create parallel image/video/VFI job tables.

Supported job types:

```text
image_upscale
video_upscale
vfi_only
engine_build
engine_validate
benchmark
```

Expand `pending_upscales` additively for project/scene/segment identity, payer/priority, requested/actual provider, enhancer worker/GPU, model/version/backend, input/settings/execution/output metadata, status/stage/progress, telemetry, attempts/errors/debug, and timestamps.

Image jobs also persist:

```text
input_width
input_height
input_aspect_ratio
input_megapixels
shape_bucket
```

### 16.3 TensorRT metadata in D1

Never put engine binaries in D1. Engine-build/history rows retain model/version, checkpoint/ONNX hash, precision, TRT/CUDA, compatibility target, compute capability/exact GPU, shape/profile, builder provider/GPU, exact `engine_r2_key`, SHA-256, size, build/validation/benchmark metadata, active state, and requesting-admin audit identity.

V1 does not require a separate TRT binary table. R2 stores the binary; D1 is the authoritative registry/control state.

---

## 17. Engine lookup and job handoff

```text
Worker resolves requested model/profile/GPU compatibility
-> query authoritative D1 engine metadata
-> choose exact/same-CC/portable active engine if valid
-> include engine identity, expected SHA/metadata and private R2 access information in the job
-> pod obtains matching .engine from R2
-> verify SHA/metadata
-> deserialize/self-test
-> execute
```

Do not proxy engine bytes through the frontend. Do not scan R2 folders and guess which engine is active; D1 supplies the exact immutable key.

If no compatible engine exists, use approved build/cache behavior or native GPU PyTorch fallback according to policy.

---

## 18. Engine generation lifecycle

Admin engine generation creates a durable `pending_upscales` row with `job_type=engine_build` and the complete engine identity/profile/compatibility target.

Use a compatible enhancer FAST pod for engine generation; no separate builder Docker image is required. Reuse a compatible idle FAST pod or provision a builder through RunPod/Novita.

Compatibility target determines the builder class:

```text
Ampere CC 8.6     -> validated sm_86 builder
Ada CC 8.9        -> validated sm_89 builder
Blackwell CC 12.0 -> validated sm_120 builder
Exact GPU         -> exact requested allowed SKU
AMPERE_PLUS       -> validated permitted Ampere-or-newer builder
```

Live provider inventory remains authoritative and all builders are filtered by the <=48 GB allowlist.

### 18.1 Temporary upload -> immutable promotion

Builder:

```text
verify trusted ONNX/checkpoint hash
build engine
deserialize/self-test
correctness inference
optional benchmark
compute SHA-256
upload to models/.engine/.tmp/{engineBuildJobId}/...
report metadata
```

Worker verifies job/pod binding, object existence, size, SHA, model/version/hash, TRT/CUDA compatibility, precision, profile, and self-test.

Then:

```text
promote/copy to immutable final R2 key
BEGIN D1 transaction
  mark new engine ready/active
  persist exact R2 key/checksum/metadata
  deactivate prior engine for same logical key
COMMIT
```

The old engine stays active until replacement is completely built, uploaded, verified, and activated. Failed builds never disable the old engine. R2 upload alone never makes an engine active.

---

## 19. Engine trust boundary

Only load engines produced by the trusted SceneBuilder engine-build path. Never accept arbitrary user/browser-supplied `.engine` files.

Every load verifies trusted R2 key, SHA-256, model/version/hash, TRT/runtime compatibility and GPU target metadata.

Admin engine actions require the existing SceneBuilder `ALLOWED_EMAILS` authorization on Worker/API. UI visibility alone is not authorization.

---

## 20. Pod API and stages

Enhancer pods expose an H3-like authenticated service contract:

```text
GET  /health
GET  /ready
GET  /capabilities
GET  /telemetry
POST /jobs
GET  /jobs/:id
POST /jobs/:id/cancel
```

Representative stages:

```text
queued
waiting_for_pod
downloading
probing
upscaling/interpolating as applicable
encoding
uploading
completed
failed
cancelled
```

All state-changing job/cancel endpoints are authenticated. Callback events are job/pod bound and replay protected.

---

## 21. Enhancer authentication/security

Use the H3 operational security philosophy while keeping enhancer identity distinct:

- separate enhancer callback endpoint;
- enhancer-specific HMAC derivation domain such as `enhancer:${workerId}`;
- timestamp/nonce replay protection;
- timing-safe validation;
- callbacks bound to expected worker + job;
- engine/admin/provider values validated against server-side allowlists.

Reuse existing repository/build secret sources already used by H3 infrastructure where appropriate, including current Docker Hub, Hetzner, Hugging Face and repository-clone credentials. Do not duplicate enhancer copies solely because enhancer shares the repository.

Build-only secrets remain build/workflow secrets and are not injected into runtime pods unless genuinely required. Runtime pods receive only required enhancer auth/control and existing R2 access values following the H3 pattern.

---

## 22. Provider provisioning/lifecycle

Mirror proven H3 provider-control behavior without sharing H3 D1 rows.

RunPod uses the equivalent of name, image, custom GPU priority/list, `gpuCount=1`, GPU compute type, compatible CUDA/driver requirement, disk, env/ports, non-interruptible/non-locked policy.

Novita uses the equivalent of name, product ID, `gpuNum`, rootfs size, image URL, env/ports, minimum compatible CUDA/driver field, and billing/network settings.

Lifecycle includes:

```text
priority dispatch
provider capability/availability filtering
idle compatible worker reuse
persistent admin idle timeout
provision timeout
job timeout
heartbeat/progress
structured debug/error
retry/provider fallback
cancel
stale allocation recovery
idle reaper
enhancer delete lock
delete verification/retry/backoff
orphan cleanup
```

Forced provider selection does not silently cross providers; `Auto` may use configured fallback policy.

---

## 23. Runtime concurrency

Allowed-email admins can set:

```text
Execution: Sequential | Parallel
Parallelism: x1 | x2 | x3 | x4 | Custom
Custom initial safety range: 1..8
Auto CUDA OOM backoff: ON
```

These are runtime controls; they do not rebuild Docker or create new engines.

### 23.1 One GPU = one active video job

Do not overlap neural inference from two separate video jobs on one GPU in V1. Queued compatible videos can reuse a warm pod sequentially.

### 23.2 Video xN

For video ESRGAN, `xN` means up to N same-size frames concurrently **inside one video**. For RIFE, `xN` means up to N frame-pair/timestep tasks concurrently inside one video. GIMM only exposes validated concurrency. Combined upscale+VFI uses xN as a stage ceiling, not N tasks in both stages simultaneously.

### 23.3 OOM backoff

```text
x4 -> x3 -> x2 -> x1
```

For Custom, decrement safely. Persist unsafe GPU/model/resolution/backend/concurrency tuples so they are not repeatedly retried. Never respond to OOM by moving neural inference to CPU.

---

## 24. Storyboard exact-shape batching

Storyboard image upscale remains native PyTorch CUDA.

Parallel/true-batch eligibility in V1 requires:

```text
same image model
same backend
same target
same exact input width
same exact input height
```

Exact W×H is required. Same aspect ratio with different dimensions is not enough. Do not resize/crop/pad solely to manufacture a batch.

Default:

```text
exact same W×H -> x4
mixed W×H      -> sequential on same warm pod where useful
OOM backoff    -> x4 -> x3 -> x2 -> x1
```

If fewer than four matching jobs are available, run the available group rather than waiting indefinitely. The Worker, not the frontend alone, is authoritative for grouping/requeueing.

---

## 25. Streaming video/codec pipeline

Do not use:

```text
video -> dump JPEG/PNG -> new model process per frame -> rebuild video
```

Use a persistent pipeline:

```text
ffprobe
-> FFmpeg/NVDEC decode when supported
-> persistent GPU model / TensorRT engine
-> bounded frame/task pipeline
-> final exact output conform
-> FFmpeg/NVENC encode by default
-> audio remux/handling
-> existing SceneBuilder media upload/writeback
```

A controlled host-memory frame pipe is acceptable if true zero-copy is impractical, but measure copy overhead and keep model/engine processes persistent.

### 25.1 Output codec/admin controls

Default enhanced master:

```text
H.265 / HEVC
NVENC / hevc_nvenc
Main10
10-bit 4:2:0 validated path
hvc1
progressive
SAR 1:1
exact requested raster/FPS
```

Video Generation Timeline admin controls:

```text
Encoder: NVENC | x265
NVENC CQ default: 16
x265 CRF default: 15
```

CQ and CRF are different rate-control systems. Validate ranges server-side. Preserve/use AAC or existing audio handling as required by current output contract. Avoid unnecessary RGB/YUV and host/device round trips.

---

## 26. Audio and duration semantics

If output duration is unchanged, preserve/remux original audio where possible. Frame interpolation alone does not inherently change playback duration.

If Director playback speed/timing is baked into the enhanced output, Director/timeline policy owns final audio timing and the same speed must not be applied twice later.

Output duration remains the selected full-master duration or authoritative Director range/duration requested by the action.

---

## 27. Observability and diagnostics

Expose/record provider/instance, GPU, compute capability, driver, CUDA, PyTorch, CuPy, TensorRT, VRAM total/used/free/peak, GPU/memory utilization, temperature/power where available, NVENC/NVDEC utilization, CPU/RAM summary, active model/backend, engine key/cache hit-or-miss, dimensions/FPS, concurrency, stage, frame/task progress, model/end-to-end FPS, elapsed time and queue depth.

Per-stage timing distinguishes:

```text
download/R2
probe/decode
preprocess
upscale inference
VFI inference
final resize/conform
encode
audio/remux
upload/R2
```

Persist compact snapshots/summary in D1 on start, stage changes, restrained active intervals, and completion/failure; do not create a high-frequency D1 time-series in V1.

Structured errors include machine code, stage, model/backend/engine, GPU/VRAM, dimensions, concurrency, CUDA/TRT/driver, retry count and bounded recent log tail.

---

## 28. Video Generation Timeline UI contract

Reuse the existing `src/components/VideoGenerationTimeline.tsx` surface. Do not create a separate normal-user enhancer page.

The current code has local/Topaz upscale choices and `interpolateTo60Fps?: boolean`; migrate to explicit upscaler + VFI model + target FPS while preserving existing Timeline/Director persistence contracts.

### 28.1 Existing surfaces

Keep/reuse:

```text
Video / Image track
  Upscale All
  Upscale Selected

Controls
  Upscale tab for selected logical Director unit
```

### 28.2 Batch Upscale modal

`Upscale All` opens a modal showing eligible video clips, already upscaled, remaining, and skipped still images.

Scope:

```text
Upscale remaining             # default
Upscale all including upscaled
```

Including already-upscaled re-runs from the authoritative pre-enhancement source, never recursively from current `upscaledVideoUrl`. Keep the previous successful master active until replacement succeeds. `Upscale Selected` uses the same logic for selected units.

### 28.3 Upscaler selector

Use one normal product dropdown:

```text
Topaz · Gaia 2
Topaz · Proteus
Real-ESRGAN · Anime
Real-ESRGAN · Real
FlashVSR
```

Output target:

```text
1080p
1440p / 2K
2160p / 4K
```

### 28.4 Explicit VFI selector

Replace boolean-only `interpolateTo60Fps` with:

```text
VFI model
Target FPS
Apply Director playback speed
```

For local Real-ESRGAN / FlashVSR:

```text
None
RIFE 4.9
GIMM-VFI-F  # only when license/product enablement permits
```

For Topaz preserve external pairings:

```text
None
Topaz Chronos where supported
Topaz Apollo where supported
```

Changing upscaler revalidates VFI; invalid pairings reset to `None`. If VFI is `None`, preserve source/master FPS and hide VFI-only controls.

When VFI is enabled, target FPS includes:

```text
30
48
60
```

30→48 and non-integer mappings use timestamp scheduling and do not create new RIFE engines.

Conceptually replace:

```ts
interpolateTo60Fps?: boolean
```

with explicit fields such as:

```ts
interpolationModel?: 'none' | 'rife-4.9' | 'gimm-vfi-f' | 'topaz-chronos' | 'topaz-apollo';
targetFps?: 30 | 48 | 60;
applyDirectorPlaybackSpeed?: boolean;
```

---

## 29. Video Generation Timeline enhancer admin UI

Admin controls live inside existing Video Generation Timeline Upscale/Enhance controls following the H3 admin pattern.

```text
email in SceneBuilder2 ALLOWED_EMAILS -> render admin section
otherwise                              -> do not render
```

Every admin API independently enforces the same `ALLOWED_EMAILS`; React hiding is not authorization. Do not add a separate enhancer-admin email variable.

### 29.1 Runtime/provider controls

Expose:

```text
Idle timeout seconds
Priority
Compute provider: Auto | RunPod | Novita
GPU target: Auto | Ampere CC8.6 | Ada CC8.9 | Blackwell CC12.0 | Exact GPU
Execution: Sequential | Parallel
Parallelism: x1 | x2 | x3 | x4 | Custom
Auto CUDA-OOM backoff
```

Exact GPU inventory comes from live allowed RunPod/Novita inventory and is filtered through the <=48 GB allowlist; do not hardcode live availability/price in Docker.

### 29.2 TensorRT engine builder

Build-model dropdown contains exactly:

```text
realesr-animevideov3
realesr-general-x4v3
RIFE 4.9
```

Never offer Storyboard image ESRGAN, GIMM-VFI-F, or FlashVSR as TensorRT build models.

Expose trusted read-only model/ONNX/checkpoint identity, FP16, compatibility target, provider, GPU family/exact GPU and model-specific shape/profile.

ESRGAN selector shows all six static shapes. RIFE selector shows 1080, 1440/2K and 2160/4K classes.

Actions:

```text
Generate engine
Validate engine
Benchmark engine
Force rebuild
Deactivate engine
Delete cached engine
View build logs
Copy R2 key
```

Statuses:

```text
queued
waiting_for_pod
building
self_test
benchmarking
uploading
verifying
ready
failed
cancelled
```

Reuse an existing compatible active engine by default.

### 29.3 Encoder/benchmark controls

Expose NVENC/x265 controls from Section 25.1. Allow controlled TensorRT-vs-PyTorch, provider, GPU-family, and NVENC-vs-x265 benchmarks. FlashVSR/GIMM may be benchmarked natively but never get TensorRT options.

Record inference/end-to-end FPS, peak VRAM, CPU utilization, wall time, output bytes and relevant quality checks.

### 29.4 Admin security

Server-side validate allowed admin identity, model, profile/shape, provider/GPU, encoder quality range, trusted baked ONNX/checkpoint, and private R2 destination. Never accept arbitrary ONNX URLs or arbitrary shell/build commands from the browser. Retain audit metadata.

---

## 30. Storyboard Upscale UI/integration

Preserve existing Storyboard user flow:

```text
Upscale Image
Upscale All
Anime / Real
2K / 4K
batch progress/status
```

```text
Anime -> RealESRGAN_x4plus_anime_6B
Real  -> RealESRGAN_x4plus
```

No VFI in Storyboard.

`Upscale All` uses local FAST enhancer pods only and zero Replicate. Where batch scope is shown, support `Upscale remaining` and `Upscale all including already-upscaled`. Including already-upscaled starts from the committed/base pre-upscale image, never recursively from current upscaled image. Keep successful output active until replacement succeeds.

Allowed-email Storyboard admin may expose provider/GPU, idle timeout/priority, x1/x2/x3/x4/custom, OOM backoff, telemetry and errors/debug. **Do not expose TensorRT engine generation for Storyboard image models.**

---

## 31. Storyboard writeback

For `image_upscale` completion:

```text
scenes[n].image = new active upscaled image
scenes[n].storyboard.thumbnail = new thumbnail when current flow produces one
preserve storyboard.originalImage
preserve storyboard.originalThumbnail
```

Do not use `isEnhanced` as image-upscale provenance. Do not create enhancer-specific scene fields. Detailed provenance stays in `pending_upscales`.

---

## 32. Director writeback and automatic still-image skip

Director completion patches only:

```text
project_video_timeline.segments[n].upscaledVideoUrl
```

Do not add enhancer provenance fields to Director JSON.

Before creating a video enhancer job:

```text
active media is video -> eligible
active media is image -> skipped_image
```

For `skipped_image`: no video enhancer job, no pod allocation, no charge, no video output, and no Storyboard mutation. Bulk progress distinguishes queued/completed/failed/skipped-image counts.

Eligibility follows the **current authoritative media being enhanced**, not merely the existence of a source image reference.

---

## 33. Completion/idempotency

```text
pod produces/uploads expected output
-> Worker verifies expected object/key ownership
-> persist pending_upscales output metadata
-> patch Storyboard OR Director authoritative state
-> mark completed
```

Retries/callback replays are idempotent. Do not create duplicate permanent references or recursively re-enhance already-enhanced assets unless the user explicitly selected include-already-upscaled.

---

## 34. Cost/charge routing

Preserve existing SceneBuilder billing/team-payer/ownership semantics.

- Charge only eligible work queued under explicit product pricing.
- `skipped_image` creates no video-enhancement charge.
- Refund/reconcile according to existing failure/cancel policy.
- Do not silently substitute a cheaper/different model for a paid requested choice.
- New local-video price points must be explicit product pricing, not inferred from provider hourly cost.

Provider/GPU routing may later use measured cost-per-completed-job as a scheduling input.

---

## 35. Layered Docker build strategy

Use the old H3-style layer/cache philosophy updated for the current model set and with **no FILM layer**.

Conceptual DAG:

```text
enhancer-smoke
  -> enhancer-base
      -> enhancer-torch
          -> enhancer-vfi-models       # RIFE 4.9 + GIMM-VFI-F
          -> enhancer-esrgan-models    # four ESRGAN sets; ONNX only for video TRT targets
              -> enhancer-fast

          -> enhancer-flashvsr-runtime
              -> enhancer-flashvsr-models
                  -> enhancer-quality  # FlashVSR + RIFE 4.9 + GIMM-VFI-F
```

Goals:

- stable CUDA/system/framework lower layers;
- model layers separate from server source;
- server source copied late;
- server edits do not invalidate multi-GB framework/model layers;
- model-family edits avoid redownloading unrelated layers where BuildKit cache permits;
- remove apt/pip/build caches from final layers;
- generated TensorRT engines are not Docker layers;
- no routine PyTorch source build;
- `torch.compile()` is not required production behavior.

---

## 36. H3 repository/build-workflow isolation

Keep enhancer in `khuzaimamussawar/minimax-h3-serverless` for now. A broader repo rename may happen later; it is not required now.

Keep existing H3 source/build paths untouched. Enhancer remains under top-level `enhancer/` with its own `.dockerignore`, `docker/`, `src/`, model/manifests, and `scripts/`.

Enhancer Docker build context is `<repo>/enhancer`, not repository root. Do not place enhancer source under H3 root `src/` or enhancer Dockerfiles under H3 root `docker/`.

Use the enhancer-specific Hetzner build workflow on this feature branch, modeled on the proven H3 Hetzner lifecycle. Do not overload mature H3 build targets. Keep H3-specific `scripts/remote_build.sh` untouched and use `enhancer/scripts/remote_build.sh` for enhancer targets.

Reuse existing repository-level Docker Hub/Hetzner/Hugging Face/repository-clone build secrets where appropriate. Use BuildKit secret mounts; never bake secrets into image layers.

Enhancer validation is path-scoped and must not trigger heavyweight H3 image builds. Validate Python/imports, shell syntax, build context/target mappings, model/checkpoint/ONNX hashes, secret handling, absence of FILM, absence of image-ESRGAN TRT targets, and FAST/QUALITY capability manifests.

### 36.1 Docker build != TensorRT engine build

```text
Docker build
  -> CPU/Hetzner build infrastructure
  -> code/runtime/models/ONNX

TensorRT engine build
  -> compatible NVIDIA GPU pod
  -> generated .engine
  -> private R2 + D1 metadata
```

Generating/replacing engines does not rebuild Docker. Provider priority, concurrency, idle timeout, engine selection, telemetry sampling and encoder CQ/CRF are runtime/control-plane settings.

---

## 37. Promotion/qualification

A Docker image merely building successfully is not enough. Qualify relevant capabilities on real provider GPUs before scheduler eligibility.

Verify CUDA, PyTorch, CuPy, TensorRT, GPU-only invariant, ESRGAN native GPU, video ESRGAN TRT build/load, RIFE 4.9 native/TRT, GIMM-VFI-F CUDA/CuPy, FlashVSR on approved QUALITY GPUs, NVENC/NVDEC where required, and R2/auth/callback behavior.

Record immutable image digest/build revision for admin diagnostics.

---

## 38. Source/license policy

Pin/review the exact source revisions and dependency licenses before commercial production.

```text
Real-ESRGAN -> use permissive official upstream; deterministic BasicSR compatibility patch
RIFE 4.9   -> use permissive upstream/source; do not copy non-commercial TRT wrapper code
FlashVSR   -> verify/pin upstream/custom-kernel dependency licenses
GIMM-VFI-F -> production blocked until required commercial permission/license is confirmed
Topaz       -> remains external; do not redistribute its assets locally
```

Community wrappers/Reddit examples are implementation references only unless their licenses permit commercial reuse.

---

## 39. Final capability matrix

```text
FAST
  CUDA 13.x
  PyTorch 2.x CUDA-13 build
  CuPy cuda13x
  TensorRT 10.16.1

  IMAGE UPSCALE
    RealESRGAN_x4plus_anime_6B -> native GPU PyTorch only
    RealESRGAN_x4plus          -> native GPU PyTorch only

  VIDEO UPSCALE
    realesr-animevideov3       -> TensorRT FP16 preferred; native GPU fallback
    realesr-general-x4v3       -> TensorRT FP16 preferred; native GPU fallback

  VFI
    RIFE 4.9                   -> TensorRT FP16 preferred; native GPU fallback
    GIMM-VFI-F                 -> native PyTorch + CuPy only; license gate

QUALITY
  CUDA 13.x
  PyTorch 2.x CUDA-13 build
  CuPy cuda13x
  TensorRT 10.16.1 for RIFE

  VIDEO UPSCALE
    FlashVSR                   -> native GPU only; no TensorRT

  VFI
    RIFE 4.9                   -> TensorRT FP16 preferred; native GPU fallback
    GIMM-VFI-F                 -> native PyTorch + CuPy only; license gate

EXTERNAL/PREMIUM
  Topaz Gaia 2 / Proteus
  Topaz Chronos/Apollo where supported
```

Locked `.engine` scope:

```text
realesr-animevideov3
realesr-general-x4v3
RIFE 4.9
```

Locked logical engine count per compatibility target:

```text
6 + 6 + 3 = 15
```

**No FILM. No Storyboard image ESRGAN TensorRT. No FlashVSR TensorRT. No GIMM TensorRT. No CPU neural fallback.**

---

## 40. 21 Aug 2026 locked decisions and FlashVSR+ reference architecture

This section supersedes any earlier wording that leaves the following items optional or undecided.

### 40.1 GIMM-VFI-F remains in the product

Keep `GIMM-VFI-F` as the quality/slower local VFI choice alongside RIFE 4.9. It remains native PyTorch + CuPy CUDA with no TensorRT engine. The commercial-license gate in Section 38 still applies and must not be bypassed.

### 40.2 Scene-cut-aware VFI is required

RIFE and GIMM VFI must be scene-cut aware.

Before scheduling interpolation across adjacent source frames, detect/consume authoritative scene boundaries. Never interpolate a synthetic in-between frame from the last frame of scene A to the first frame of scene B.

At a hard cut:

```text
finish/select the previous scene endpoint according to timestamp policy
-> do not interpolate across the boundary
-> restart VFI context on the first frame of the next scene
-> preserve exact requested output timestamps and duration
```

Scene detection may also provide natural QUALITY/FlashVSR chunk boundaries, but scene splitting is orchestration; it does not create another model or TensorRT engine.

### 40.3 Do not deduplicate or skip static/duplicate frames

Do **not** add a static-frame detector that drops, collapses, skips, or substitutes duplicate-looking source frames to save VFI compute.

Preserve every authoritative source frame/timestamp in scheduling. Scene-cut handling is separate from duplicate/static-frame optimization. If a third-party RIFE/GIMM wrapper exposes a `static-skip`, duplicate-frame skip, or equivalent optimization, keep it disabled unless a future explicit product decision reverses this rule.

### 40.4 AMPERE_PLUS portable TensorRT set is retained

`AMPERE_PLUS` is a retained first-class portable fallback set, not merely an experiment.

Baseline prewarmed cache target when all architecture sets are released:

```text
sm_86 same-CC set      = 15 engines
sm_89 same-CC set      = 15 engines
sm_120 same-CC set     = 15 engines
AMPERE_PLUS portable   = 15 engines
-------------------------------------
baseline cache         = 60 engines
```

Exact-GPU sets remain optional and add another 15 only when deliberately generated/benchmarked.

Runtime preference stays:

```text
exact GPU
-> same compute capability
-> AMPERE_PLUS portable
-> native GPU PyTorch fallback
```

### 40.5 Locked local video pricing

Use existing SceneBuilder payer/team/credit/refund/idempotency mechanics. Do not create another wallet or billing subsystem.

Local video enhancement pricing is:

```text
FAST + no VFI or RIFE 4.9   = 3 credits / billable second
FAST + GIMM-VFI-F           = 5 credits / billable second
QUALITY / FlashVSR          = 10 credits / billable second
```

`QUALITY = 10 credits/second` is the local QUALITY price regardless of whether its allowed local VFI choice is None, RIFE, or GIMM, unless a future explicit pricing decision changes it.

Rules:

- `skipped_image` = 0 credits;
- callback/retry replay never charges twice;
- failed/cancelled work follows existing SceneBuilder refund/reconciliation behavior;
- Topaz keeps its existing external/premium pricing path;
- billable duration is the authoritative media range actually selected for enhancement;
- when `Apply Director playback speed` is enabled and that speed is baked into the requested enhanced result, use the resulting authoritative Director duration/range for billing;
- otherwise use the selected source/master range duration;
- use the existing SceneBuilder credit normalization/rounding convention rather than inventing a second rounding rule.

### 40.6 Approved repository rename target and audit

The approved future GitHub repository name is:

```text
khuzaimamussawar/GPU-runtime
```

The rename is a repository-name migration only. It must not rename H3 Docker Hub images, D1 tables, H3 service names, runtime APIs, or mature H3 lifecycle identifiers.

Audit findings before rename:

- the maintained H3 Hetzner build workflow passes GitHub's dynamic `GITHUB_REPOSITORY` into the remote build, so cloning follows the renamed repository automatically;
- the H3 `scripts/remote_build.sh` uses `${GITHUB_REPOSITORY}` for the actual Git clone URL;
- `/opt/minimax-h3-serverless` in the H3 remote builder is only a temporary local checkout-directory name and may remain initially;
- the existing secret name `GH_FH_TOKEN_MM_H3_SERVERLESS` may remain initially; renaming that secret is a separate migration and is not required for the repository rename;
- enhancer Hetzner build workflow/remote builder also use dynamic repository identity and are compatible with the rename;
- SceneBuilder2 uses `minimax-h3-*` primarily as Docker/runtime image/service naming, not as a hardcoded GitHub clone dependency; those runtime image names remain unchanged;
- no external `uses: khuzaimamussawar/minimax-h3-serverless/...@...` GitHub Action reference was found in the audited H3/SceneBuilder2 repositories.

After the GitHub repository rename, update documentation links and developer/local git remotes. Do not rebuild heavyweight H3 Docker images merely because the GitHub repository name changed.

### 40.7 SECourses FlashVSR+ is an implementation reference, not a ComfyUI dependency

SECourses Upscaler Pro is a custom Python/Gradio application with its own installer, queue, health checks and model orchestration. The supplied tutorial demonstrates FlashVSR+ running directly in this custom app on Windows and also describes cloud use. It is not presented as a ComfyUI workflow.

The current public SECourses post says the Upscaler application has moved to Torch 2.13 + CUDA 13 with current compiled libraries. The same post lists CUDA 13 and cuDNN 9.17+ in Windows requirements. The exact CUDA 13 minor used by the current FlashVSR+ backend is not publicly locked in that post, so do not infer a precise minor from it.

The tutorial's separate statement about CUDA 13 + Torch 2.9.1 occurs in the later Trellis/3D-library section and must **not** be treated as the FlashVSR+ runtime version.

SECourses says that on 20 Feb 2026 he completely changed FlashVSR+ to a new repository and significantly modified it, but the public post does not identify that exact repository. Do not claim a specific public fork is his exact backend without direct evidence.

### 40.8 FlashVSR+ behavior worth adopting/qualifying for SceneBuilder QUALITY

SECourses' demonstrated FlashVSR+ architecture uses or exposes these concepts:

```text
FlashVSR v1.1 / FlashVSR+ native GPU inference
2x or 4x model scale
optional pre-downscale to land on the desired final output raster
scene detection
scene/chunk-based processing
resume at completed scene/chunk boundaries
GPU-dependent frame chunk length
DiT tiling for VRAM control
large tile / fewer-tile preference while staying below OOM/shared VRAM
Tile Overlap adjustment if seams appear
VAE tiling generally avoided in the full-model path when it causes quality/noise issues
RIFE as a separate interpolation stage
FFmpeg audio preservation
H.265 / 10-bit output controls
queue/cancel/health/progress/metadata
```

For SceneBuilder QUALITY:

- keep FlashVSR native GPU only and no TensorRT;
- scene/chunk processing is approved and is compatible with our required scene-cut-aware VFI;
- for our <=15-second, typically ~24 FPS clips, do not load the whole product architecture around multi-hour resume, but make scene/chunk boundaries restartable/idempotent so a failed QUALITY job can resume/retry safely where practical;
- do not copy SECourses proprietary code; reproduce permitted architecture from open/permissive upstreams;
- FlashVSR-specific **DiT tiling is allowed** as VRAM control and does not change the separate rule that ESRGAN and RIFE remain non-tiled;
- on the QUALITY floor (4090-class or better), prefer non-DiT-tiled/full execution when it passes VRAM and quality tests; otherwise use the largest validated DiT tile that stays out of OOM/shared-memory spill;
- keep VAE tiling disabled by default unless our own tests show a valid case;
- choose frame chunk length from measured GPU/output-resolution headroom rather than hardcoding one value for every GPU;
- preserve audio and exact duration when merging chunks;
- do not use shared system RAM as a hidden substitute for insufficient VRAM.

### 40.9 Public FlashVSR+ candidate architecture to evaluate

A public Apache-2.0 FlashVSR+ lineage (`lihaoyun6/FlashVSR_plus`, and forks based on it) is technically relevant because it explicitly supports CUDA 12.8 or CUDA 13.0 PyTorch installs, introduces Blackwell support, replaces the original Block-Sparse-Attention build with Sparse SageAttention, adds DiT tiling/memory optimizations, provides a direct Gradio/CLI runtime, and copies audio through FFmpeg.

This public lineage is **not confirmed to be SECourses' exact private/custom backend**. Treat it as a candidate implementation/reference only.

Before selecting it for SceneBuilder QUALITY, compare against official FlashVSR v1.1 and verify:

```text
v1.1 model fidelity
locality-constrained sparse-attention behavior / no dense-attention quality regression
Ampere sm_86
Ada sm_89
Blackwell sm_120
CUDA 13 candidate stack
PyTorch 2.13 candidate stack
4090/48GB/5090 VRAM and throughput
DiT tiled vs non-tiled quality
scene-boundary continuity
exact frame count / duration / audio sync
```

Official FlashVSR has warned that third-party implementations that omit its LCSA behavior can degrade quality, so SceneBuilder must qualify actual output rather than assuming every `FlashVSR+` fork is equivalent.

### 40.10 Current CUDA/PyTorch qualification candidate

Current facts:

```text
SECourses current Upscaler post -> Torch 2.13 + CUDA 13
PyTorch stable wheel exists     -> torch 2.13.0 + cu130
CuPy                            -> cupy-cuda13x
TensorRT                        -> 10.16.1
```

Do not reinterpret `cu130` as PyTorch version 13; `2.13.0` is the PyTorch version and `cu130` is its CUDA build tag.

For SceneBuilder, qualify the exact system-CUDA minor against **all** FAST + QUALITY dependencies before changing Section 4 from `13.x` to a precise minor. TensorRT 10.16.1 supports the CUDA 13 family and NVIDIA notes a CUDA-13.0 edge-Blackwell issue that is fixed in CUDA 13.1, while the normal stable PyTorch 2.13 wheel is published as `cu130`. Therefore the final minor pin must be chosen by real Ampere/Ada/Blackwell integration tests, not by assuming SECourses' unspecified CUDA-13 minor.

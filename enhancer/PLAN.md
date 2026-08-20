# SceneBuilder Enhancer Implementation Plan

Status: **canonical implementation plan**

Branch: `feat/enhancer-gpu-runtime`

This file is intentionally separate from the H3 runtime/lifecycle design. It consolidates the approved enhancer decisions from the older SceneBuilder enhancement plans plus the newer CUDA/TensorRT/GIMM decisions. When older drafts conflict with this file, this file wins.

---

## 1. Product goal

Build a SceneBuilder image/video enhancer that uses RunPod and Novita GPU pods with the same operational discipline as the existing H3 pod system while remaining physically isolated from H3 state.

Required product capabilities:

- Storyboard single-image upscale through the existing legacy path unless changed later.
- Storyboard **Upscale All** through local enhancer pods only; zero Replicate for that batch path.
- Video Generation Timeline / Director video upscale.
- FAST and QUALITY local GPU enhancement modes.
- Exact output targets: 1080p, 1440p and 2160p/4K for video; existing 2K/4K Storyboard targets for images.
- Explicit frame interpolation model and target FPS.
- RIFE 4.9 as the fast/default local VFI.
- GIMM-VFI-F as the slower/high-quality local VFI where licensing permits production use.
- Existing Topaz premium/external options remain available through their external backend.
- Existing SceneBuilder R2 media ownership/layout and existing Storyboard/Director writeback contracts stay intact.
- H3-style priority, retries, error/debug state, heartbeats, idle reuse, configurable idle timeout, provision/job timeout, provider fallback, delete locks, delete verification/backoff and orphan cleanup.
- Neural inference must never silently run on CPU.

The expensive spatial enhancement should normally create an enhanced master once. Trim/speed/FPS changes must not automatically force another ESRGAN/FlashVSR spatial pass when only a VFI/timing rerun is required.

---

## 2. Hard non-negotiables

### 2.1 H3 lifecycle remains untouched

H3 state remains:

```text
h3_pod_workers
video_generation_batches
video_generation_jobs
```

Enhancer state remains separate:

```text
enhancer_pod_workers
pending_upscales
enhancer_pod_delete_locks
enhancer_event_nonces
enhancer_config
```

Rules:

- Never put enhancer rows into `h3_pod_workers`.
- H3 code never queries `enhancer_pod_workers`.
- Enhancer code never treats `h3_pod_workers` as an enhancer worker pool.
- Enhancer uses its own delete locks, callbacks, replay protection and lifecycle state.
- Preserve existing H3 runtime/lifecycle source paths and behavior.
- Shared provider/R2 helpers may be reused only when they are stateless with respect to H3 lifecycle state.

### 2.2 Product-state D1 schema remains unchanged

Do not add enhancer-specific SQL/product schema to:

```text
projects_timeline
project_video_timeline
```

Do not create a second project media-state table.

Detailed provider/model/GPU/engine/progress/error provenance belongs in operational enhancer state, primarily `pending_upscales` and `enhancer_pod_workers`.

### 2.3 Existing permanent R2 media layout remains intact

Permanent product outputs continue to resolve through SceneBuilder's existing media-layout logic. Canonical V2 semantic destinations remain:

```text
Storyboard image upscale:
projects/{projectId}/images/scenes/upscaled/

Director video upscale:
projects/{projectId}/video/upscaled/
```

Do not perform an R2 media migration as part of enhancer work.

### 2.4 Replicate boundary

- Storyboard `POST /api/upscale-batch` / **Upscale All** -> enhancer pods only.
- No Replicate cron/worker may claim local enhancer batch rows.
- New local batch jobs start with an enhancer-owned state such as `waiting_for_pod`, not the legacy Replicate-claimable state.
- Existing single-image `/api/upscale` remains the legacy Replicate path unless deliberately changed later.

---

## 3. Final model/product matrix

### 3.1 Local Storyboard image upscalers

```text
Anime -> RealESRGAN_x4plus_anime_6B
Real  -> RealESRGAN_x4plus
```

Rules:

- native PyTorch CUDA only;
- no TensorRT `.engine` files;
- no ONNX/TensorRT requirement;
- no VFI in Storyboard image upscale.

### 3.2 Local Director FAST video upscalers

```text
Anime -> realesr-animevideov3
Real  -> realesr-general-x4v3
```

Rules:

- TensorRT FP16 preferred when a validated matching engine exists;
- native full-frame PyTorch CUDA fallback only;
- never CPU;
- no tiled ESRGAN execution in the approved V1 path.

`realesr-general-x4v3` uses one fixed balanced denoise preset for V1. Do not generate extra TensorRT engines for a continuum of DNI/denoise values. If product needs multiple denoise choices later, add a small explicit preset set only after measurement.

### 3.3 Local QUALITY video upscaler

```text
FlashVSR v1.1
```

Rules:

- QUALITY image only;
- native CUDA/PyTorch/custom-attention path;
- no TensorRT `.engine`;
- no CPU fallback;
- architecture support only after actual model self-test on that runtime/GPU combination.

### 3.4 Local VFI

```text
RIFE 4.9    -> fast/default; TensorRT where validated
GIMM-VFI-F  -> quality/slower; native PyTorch + CuPy only
```

Do not substitute GIMM-VFI-R or perceptual `F-P` variants without a future product decision.

### 3.5 External premium models remain supported

Keep the existing external premium backend contract for:

```text
Topaz Proteus
Topaz Gaia 2
Topaz Chronos where supported
Topaz Apollo where supported
```

Topaz rules:

- external/provider-specific backend;
- not baked into enhancer Docker;
- no SceneBuilder local TensorRT engine generation;
- no entries in the local TensorRT R2 engine cache;
- keep licensing/provider handling separate from local open-source model packaging;
- keep user-facing Video Generation Timeline options as described later in this plan.

### 3.6 Removed model

**FILM is not part of the enhancer product/runtime.** Do not install, expose, enqueue, advertise, benchmark, create a Docker layer for, or create a TensorRT artifact for FILM.

---

## 4. Runtime topology

There are exactly two local enhancer GPU image families:

```text
scenebuilder-enhancer-fast
scenebuilder-enhancer-quality
```

### 4.1 FAST

```text
one CUDA 13.x system userspace line
PyTorch 2.x CUDA-13 build selected by qualification
CuPy: cupy-cuda13x
TensorRT 10.16.1 runtime + builder
FFmpeg with NVDEC/NVENC support
NVML telemetry

Models:
  RealESRGAN_x4plus_anime_6B
  RealESRGAN_x4plus
  realesr-animevideov3
  realesr-general-x4v3
  RIFE 4.9
  GIMM-VFI-F
```

### 4.2 QUALITY

```text
one CUDA 13.x system userspace line
PyTorch 2.x CUDA-13 build selected by qualification
CuPy: cupy-cuda13x
TensorRT 10.16.1 runtime for RIFE engine execution
FFmpeg with NVDEC/NVENC support
NVML telemetry

Models:
  FlashVSR v1.1
  RIFE 4.9
  GIMM-VFI-F
```

### 4.3 One CUDA line per Docker image

Do not install CUDA 11 + CUDA 12.x + CUDA 13 side-by-side.

`PyTorch 2.x` is the PyTorch release version. A tag such as `cu130` is only the CUDA build/runtime tag for that PyTorch distribution; it is not a PyTorch major version.

`cupy-cuda13x` is the CuPy package for the CUDA 13 family; do not install `cupy-cuda12x` beside it.

The exact CUDA 13 minor and exact PyTorch 2.x pin are qualification locks, not reasons to install a second system CUDA toolkit in the same image.

---

## 5. GPU architecture and fleet policy

Supported architecture families:

```text
Ampere    -> primarily sm_86 class for this fleet
Ada       -> sm_89
Blackwell -> sm_120 RTX/workstation class
```

Global product rule:

```text
VRAM <= 48 GB, including 48 GB GPUs
```

Do not provision A100/A800/H100/H200/B200-class workers or other excluded high-cost datacenter SKUs even when a particular variant technically fits the VRAM threshold.

Representative allowed candidates, subject to live provider inventory and model self-test:

```text
Ampere:
  RTX A4000
  RTX A4500
  RTX A5000
  RTX 3090
  A40 (48 GB)
  RTX A6000 (48 GB)

Ada:
  RTX 2000 Ada
  RTX 4000 Ada
  L4
  RTX 4090
  L40 (48 GB)
  L40S (48 GB)
  RTX 6000 Ada (48 GB)

Blackwell <= 48 GB:
  RTX 5090
  RTX PRO workstation variants <= 48 GB when offered/validated
```

Live provider inventory is authoritative. Provider aliases must be normalized to canonical GPU names in the control plane.

---

## 6. GPU-only execution invariant

Neural inference must never silently fall back to CPU.

```text
torch.cuda.is_available() == false
  -> worker unhealthy
  -> never advertise ready
  -> report error
  -> delete/replace pod

model parameters on CPU at neural inference boundary
  -> abort/requeue job

input tensors on CPU at neural inference boundary
  -> abort/requeue job

unexpected model offload to CPU
  -> capability/job failure
```

Do not inherit ComfyUI automatic CPU/offload semantics in production.

The worker owns:

```text
CUDA_VISIBLE_DEVICES
cuda:0 selection
model.to(cuda:0)
input tensor placement
output validation
CUDA synchronization/events
```

CPU remains valid for orchestration, HTTP, R2 I/O, JSON, filesystem work, ffprobe, some decode/encode/pre/post processing and process supervision. The neural models stay GPU-backed.

### 6.1 Startup qualification

Before `/ready` is true, run real checks.

FAST:

```text
CUDA allocation + CUDA op
CuPy CUDA kernel
image ESRGAN native CUDA inference
video ESRGAN native CUDA inference
available video ESRGAN TensorRT deserialize/inference tests
RIFE 4.9 native CUDA inference
available RIFE TensorRT deserialize/inference test
GIMM-VFI-F native CUDA/CuPy inference
FFmpeg/NVENC smoke test when NVENC is configured
```

QUALITY:

```text
CUDA allocation + CUDA op
CuPy CUDA kernel
FlashVSR representative inference
RIFE 4.9 native CUDA inference
available RIFE TensorRT test
GIMM-VFI-F native CUDA/CuPy inference
FFmpeg/NVENC smoke test when configured
```

Record GPU name, VRAM, compute capability, driver, CUDA, PyTorch, CuPy, TensorRT, model pass/fail, peak VRAM and NVML telemetry.

`/capabilities` advertises only models/backends that actually passed.

---

## 7. Docker layering and repository/build architecture

Enhancer stays in the existing H3 runtime repository for now under its own top-level tree:

```text
enhancer/
  .dockerignore
  docker/
  src/
  models/
  scripts/
```

Do not move enhancer code into root H3 `src/` or enhancer Dockerfiles into root H3 `docker/`.

Keep existing H3 source, Dockerfiles, workflows and H3 `scripts/remote_build.sh` unchanged.

A future repository rename to a broader SceneBuilder GPU-runtime name is allowed as a separate migration. Do not require that rename to implement enhancer.

### 7.1 Layered enhancer Docker DAG

Use layered builds so changing server code does not rebuild/download every heavy model/runtime layer.

Recommended current targets, with **no FILM layer**:

```text
Dockerfile.smoke
Dockerfile.base
Dockerfile.torch
Dockerfile.esrgan-models
Dockerfile.rife-models
Dockerfile.gimm-models
Dockerfile.fast-models
Dockerfile.fast
Dockerfile.flashvsr-runtime
Dockerfile.flashvsr-models
Dockerfile.quality
```

Principles:

- one selected CUDA userspace line per final image;
- prebuilt PyTorch CUDA packages rather than building PyTorch from source;
- `torch.compile()` is not required for production V1;
- model/checkpoint layers before application/server code;
- remove pip/apt/build caches from final images;
- multi-stage build where useful;
- generated TensorRT engines are never baked into Docker.

### 7.2 Enhancer build workflow

Keep the proven H3 workflow untouched and add/maintain an enhancer-specific Hetzner workflow modeled on it, e.g.:

```text
.github/workflows/hetzner-enhancer-build.yml
```

Enhancer workflow semantics should mirror H3's proven temporary-builder lifecycle:

```text
workflow_dispatch
-> create temporary Hetzner builder
-> wait for SSH
-> clone current repository/ref
-> build selected enhancer target(s)
-> push Docker image
-> delete builder on success
-> optional keep-on-failure debug window
```

Use `enhancer/scripts/remote_build.sh`; do not add enhancer target cases to H3's root `scripts/remote_build.sh`.

Enhancer Docker build context is exactly:

```text
repository/enhancer
```

not the full repository.

### 7.3 Existing repository secrets

Reuse existing repository build secrets; do not duplicate values under enhancer-specific names without a separate migration:

```text
DOCKERHUB_TOKEN
DOCKERHUB_USERNAME
HETZNER_TOKEN
HF_TOKEN
GH_FH_TOKEN_MM_H3_SERVERLESS
```

Secrets are build/runtime inputs only; never bake secret values into image layers.

### 7.4 Enhancer validation workflow

Maintain a separate enhancer validation workflow that checks at minimum:

```text
Python compile/import
shell syntax for enhancer/scripts/remote_build.sh
model/checkpoint checksum manifest
Docker target lineage/mapping
no prohibited secret ENV baking
Docker COPY paths remain inside enhancer build context
no accidental H3 source/Docker modifications
```

---

## 8. Artifact ownership: Docker vs R2 vs D1

### 8.1 Docker owns durable model sources

Docker contains the model/runtime source artifacts required for operation:

```text
model code
weights/checkpoints (.pth/.ckpt/.safetensors as applicable)
video ESRGAN ONNX used for TensorRT builds
RIFE 4.9 ONNX used for TensorRT builds
runtime dependencies
TensorRT builder/runtime
FFmpeg/NVENC/NVDEC helpers
NVML telemetry
HTTP service
```

Do not require a network volume for model weights.

The two image ESRGAN models do not need ONNX/TensorRT assets because they are native PyTorch-only by policy.

### 8.2 Private R2 owns generated TensorRT artifacts

Generated local TensorRT artifacts live in private R2:

```text
.engine/.plan binaries
TensorRT timing caches when useful
optional immutable diagnostic sidecar JSON
```

No model-weight migration to R2 is required.

Product media continues to use the existing SceneBuilder R2 media paths from Section 2.3. The special `models/.engine/` tree is only for generated TensorRT artifacts.

### 8.3 D1 owns metadata/control state, not binaries

Never put an engine binary in D1.

V1 uses the existing `pending_upscales` table for enhancer jobs **including engine-build/validate/benchmark history and active engine metadata** rather than introducing an additional `enhancement_trt_artifacts` table unless real query/load evidence later justifies one.

D1 stores exact R2 engine key/checksum/compatibility/activation metadata.

### 8.4 Engine transport to a pod

A job that needs TensorRT includes the selected engine artifact identity in the signed job payload:

```text
engine key/id
exact R2 object key
SHA-256
size
model/version
TensorRT/CUDA compatibility metadata
shape/profile
short-lived signed R2 download URL or equivalent trusted R2 access
```

Do not base64/embed large `.engine` binaries in the control-plane JSON request.

The pod downloads the exact selected engine into pod-local ephemeral working storage (for example `/cache/engines`) and validates it before deserialize. That local copy is disposable and is never a source of truth.

---

## 9. TensorRT version and scope

### 9.1 TensorRT version

Pin **TensorRT 10.16.1** for the current phase.

Do not move to TensorRT 11.x until our own exporters/runtime are deliberately migrated and parity-tested.

### 9.2 Exactly three TensorRT model families

Only:

```text
1. realesr-animevideov3
2. realesr-general-x4v3
3. RIFE 4.9
```

Explicitly no `.engine` for:

```text
RealESRGAN_x4plus_anime_6B
RealESRGAN_x4plus
FlashVSR
GIMM-VFI-F
```

---

## 10. Exact TensorRT engine topology

### 10.1 Video ESRGAN static full-frame engines

Use static full-frame TensorRT for the two video ESRGAN models. Do **not** make tiled TensorRT the primary path and do not use tiled ESRGAN as an OOM fallback in the approved design.

Canonical landscape source rasters:

```text
854x480
1056x594
1280x720
```

Portrait mirrors:

```text
480x854
594x1056
720x1280
```

Per video ESRGAN model:

```text
3 landscape + 3 portrait = 6 engine files
```

For the two video models:

```text
realesr-animevideov3   = 6
realesr-general-x4v3   = 6
--------------------------------
video ESRGAN total     = 12
```

### 10.2 RIFE 4.9 engines: three files, not six

Initial RIFE TensorRT set per compatibility target is exactly **3 engine files**:

```text
1080 class
  profile A: 1920x1080
  profile B: 1080x1920

1440 class
  profile A: 2560x1440
  profile B: 1440x2560

2160/4K class
  profile A: 3840x2160
  profile B: 2160x3840
```

Landscape and portrait are optimization profiles inside each resolution-class engine; they do not create six files unless later benchmarks prove profile splitting is materially better.

RIFE is full-frame motion interpolation. Do not tile RIFE.

### 10.3 Total engine count

Per TensorRT compatibility target:

```text
12 video ESRGAN engines
 3 RIFE engines
-----------------------
15 engine files
```

This count is **not** per clip, duration, target FPS, or runtime xN concurrency.

If we build architecture-specific sets for all three current compute-capability families:

```text
sm_86 Ampere    -> 15
sm_89 Ada       -> 15
sm_120 Blackwell-> 15
----------------------
family-specific total -> 45 cached engine files
```

An optional `AMPERE_PLUS` portable set would add another 15. Each deliberately generated exact-GPU set would add another 15.

Do not create those optional sets blindly; benchmark first.

### 10.4 What never creates another engine

No new engine for:

```text
video duration
number of frames
24 -> 48 FPS
24 -> 60 FPS
30 -> 48 FPS
30 -> 60 FPS
RIFE x2/x3/x4 task concurrency
ESRGAN x2/x3/x4 frame concurrency
trim length
Director playback-speed checkbox
```

Those are runtime scheduling/timestamp concerns.

---

## 11. Video ESRGAN source-raster normalization

This is enhancer input normalization only. **It does not put ESRGAN/RIFE/TensorRT inside H3 pods or H3 Serverless.**

H3 may produce recurring delivery rasters that the separate enhancer later receives as ordinary input media.

Known H3-origin landscape mappings:

```text
.4 MP -> 854x480  -> exact 480 engine
.6 MP -> 1056x594 -> exact .6 engine
.7 MP -> 1138x640 -> normalize to 1056x594 when the quality rule allows
.9 MP -> 1280x720 -> exact 720 engine
```

Portrait mirrors use the corresponding portrait engines.

### 11.1 Never stretch

For small aspect-ratio mismatch, crop the small excess first, then resize with a high-quality Lanczos filter.

Example for a 16:9 project:

```text
1344x768
-> center crop to 1344x756
-> Lanczos resize to 1280x720
-> static video ESRGAN TensorRT engine
```

Never stretch `1344x768` directly to `1280x720`.

Initial small automatic aspect-mismatch tolerance may use approximately 3% as in the older plan; larger mismatches must not silently destroy composition.

### 11.2 Non-canonical fallback

Do not throw away major source detail just to hit a TensorRT shape.

Example:

```text
1920x1080 upload + no matching video ESRGAN engine
-> native full-frame PyTorch CUDA FP16 at the appropriate source raster
-> never CPU
```

Do not downscale 1080p to 720p solely to force an engine match.

If full-frame PyTorch cannot fit, lower runtime concurrency and/or requeue to a larger allowed VRAM GPU. The approved design does not introduce ESRGAN tiling as a hidden fallback.

### 11.3 Neural scale vs delivery resolution

The ESRGAN networks are native x4 models. Neural scale and final delivery target are separate:

```text
source
-> x4 neural inference
-> high-quality exact final resize/crop
-> requested 1080p / 1440p / 4K
```

Final target resolution does not multiply the number of TensorRT engine files.

---

## 12. RIFE 4.9 timestamp/FPS/duration logic

RIFE must support arbitrary interpolation timestamps; do not implement it only as integer `--multi` x2/x3/x4.

For source FPS `S`, output target FPS `T`, and effective playback speed `p`:

```text
sourcePositionStep = S * p / T
```

For each output time:

```text
sourcePosition = outputTime * S * p
left  = floor(sourcePosition)
right = left + 1
t     = sourcePosition - left
```

Call RIFE using the source frame pair and fractional timestep `t`.

Examples using the same RIFE engine:

```text
24 -> 48
24 -> 60
30 -> 48
30 -> 60
```

No new engine is required for any of those mappings.

### 12.1 Trim/duration semantics

The VFI scheduler must operate on the authoritative selected media range.

If the master video is 10 seconds but the Director/user selects a 6-second effective segment/range, process the selected authoritative range according to the existing Director trim semantics rather than blindly interpolating ten seconds and presenting it as six.

### 12.2 Director playback-speed checkbox

When the user enables the `Apply Director playback speed` option, VFI scheduling incorporates the current Director speed and the resulting VFI master represents that timing.

When disabled, VFI operates at normal source/master timing and Director speed stays a later TimelineRender operation.

Do not apply the same speed twice.

If only Director speed or requested FPS later changes and the spatially enhanced master is still valid, rerun only VFI/timing from the spatially enhanced master; do not rerun ESRGAN/FlashVSR unnecessarily.

---

## 13. GIMM-VFI-F

Selected local quality VFI:

```text
GIMM-VFI-F
FlowFormer flow estimator
native PyTorch + CUDA/CuPy
```

No ONNX/TensorRT `.engine` in this phase.

### 13.1 License blocker

The original GIMM-VFI licensing is non-commercial by default unless commercial permission is obtained. Production enablement remains blocked until SceneBuilder has the required commercial permission/license.

Testing/integration must respect the upstream license.

---

## 14. FlashVSR

FlashVSR remains the QUALITY spatial upscaler.

Rules:

- native CUDA/PyTorch/custom attention;
- no TensorRT;
- no CPU fallback;
- model-specific temporal/spatial chunking may be used as required by FlashVSR's implementation;
- do not require the entire clip to reside in VRAM simultaneously;
- capability-gate each actual GPU/runtime combination with a real startup inference test;
- current production target clips are short (normally <=15 seconds), but streaming/chunking still prevents unnecessary peak VRAM and keeps runtime stable.

QUALITY GPU selection should start from the validated 24 GB / RTX-4090-class-or-better pool and may use 32/48 GB candidates when required by measured peak VRAM. Exact eligibility is self-test/benchmark driven.

---

## 15. Video processing/codec/audio contract

### 15.1 Processing order

Normal local path:

```text
input authoritative media
-> ffprobe
-> decode
-> trim/range/timestamp preparation
-> small aspect conform / canonical normalization if required
-> spatial upscale
-> optional VFI
-> exact final resize/crop
-> encode
-> audio remux/timeline handling
-> upload to existing SceneBuilder R2 destination
-> authenticated completion callback
```

### 15.2 Persistent streaming pipeline

Never implement video ESRGAN as:

```text
dump thousands of JPEG/PNG files
-> launch a new model process for every frame
-> rebuild video later
```

Use a long-lived worker/model/engine:

```text
FFmpeg/NVDEC where possible
-> bounded frame buffers
-> persistent GPU inference
-> FFmpeg/NVENC where possible
```

A controlled host-memory frame pipe is acceptable when true zero-copy is not practical; measure it rather than spawning per-frame processes.

### 15.3 Authoritative source

Use the highest-quality current authoritative media chosen by existing SceneBuilder/Director selection logic. Never recursively upscale the current enhanced output unless the user explicitly selects an `including already-upscaled` replacement operation, and even then start from the authoritative pre-enhancement/base source rather than using the old enhanced master as the neural input.

### 15.4 Output codec controls

Default local enhanced master target:

```text
Codec: H.265 / HEVC
Encoder: NVENC
Profile: Main10
Pixel format: 10-bit 4:2:0 / p010-style path
Container tag: hvc1
Progressive
SAR: 1:1
exact output resolution
exact requested FPS
```

Admin encoder controls:

```text
NVENC
  quality control: CQ
  default: 16
  practical admin range: 12-24

x265
  quality control: CRF
  default: 15
  practical admin range: 12-30
```

`CQ 16` and `CRF 15` are not numerically equivalent quality scales.

x265 is allowed as CPU **encoding** for controlled quality/compression comparisons; this does not permit CPU neural inference.

Where supported/validated, prefer high-quality NVENC settings such as P7/UHQ/full-resolution multipass.

### 15.5 TimelineRender smart-copy compatibility

Produce enhanced masters compatible with existing TimelineRender H.265 expectations so unchanged compatible clips can use stream copy.

When pixels/timing must change, TimelineRender decodes/filters/re-encodes normally.

### 15.6 Audio

If duration/timing is unchanged, preserve/remux original audio where possible.

VFI alone does not inherently change duration.

If Director playback speed is baked into the new VFI master, mark/handle that through the existing Director/timeline timing contract so audio/timing are not applied twice.

### 15.7 Variable frame rate

Use `ffprobe` timestamps rather than blindly assuming every source is perfect CFR. Preserve the intended authoritative duration and final frame/timestamp contract.

For the Director playback-speed/range option, the requested effective Director duration becomes the VFI timing target when that option is enabled.

---

## 16. D1 operational schema

### 16.1 `enhancer_pod_workers`

Dedicated enhancer worker lifecycle table. Suggested required ownership:

```text
id
service_kind                # enhancer_fast | enhancer_quality
project_id/user_email where relevant
provider
provider_instance_id
endpoint_url
gpu_class
provider_gpu_name
region
status
current_job_id
runtime_image
runtime_image_digest
cuda_version
pytorch_version
cupy_version
tensorrt_version
capabilities_json
telemetry_json
last_heartbeat_at
last_error_code
last_error_json
idle_since
idle_timeout_seconds
terminate_after
last_used_at
delete_retry_count
delete_retry_after
created_at/updated_at/... timestamps
```

### 16.2 Unified job table: existing `pending_upscales`

Do not create separate tables for image upscale, video upscale, VFI, engine build, validation and benchmarks.

Expand the existing physical table in place while preserving current Storyboard status/cancel/cleanup compatibility.

Job types:

```text
image_upscale
video_upscale
vfi_only
engine_build
engine_validate
benchmark
```

Suggested enhancer fields include:

```text
id
job_type
project_id
scene_id / segment_id
user_email
priority

provider_requested
provider_actual
pod_worker_id
gpu_class
provider_gpu_name

model_family
model_version
backend

engine_key
engine_r2_key
engine_sha256
engine_size_bytes
engine_active
engine_profile
engine_compatibility_mode
engine_compute_capability
engine_precision
engine_trt_version
engine_cuda_version
onnx_sha256
builder_config_hash

input_json
settings_json
execution_json
output_json

input_width
input_height
input_aspect_ratio
input_megapixels
shape_bucket

status
stage
progress
telemetry_summary_json
attempt_log_json
error_code
error_json

created_at
started_at
completed_at
updated_at
invalidated_at
```

Engine-build rows are durable registry/history records. A completed active engine is selected from indexed D1 metadata; D1 stores the exact R2 object key/checksum. No separate TensorRT registry table is required in V1.

### 16.3 Other enhancer tables

Keep separate:

```text
enhancer_pod_delete_locks
enhancer_event_nonces
enhancer_config
```

`enhancer_config` persists at least `idle_timeout_seconds` and other approved runtime/admin defaults.

---

## 17. Storyboard writeback

For `image_upscale`:

```text
scenes[n].image = new active upscaled image
storyboard.thumbnail = new active thumbnail when produced
storyboard.originalImage = preserved
storyboard.originalThumbnail = preserved
```

Do not create/overwrite Storyboard prompt-enhancement `isEnhanced` as image-upscale provenance.

Do not add `scene.enhancement`, `enhancedImageUrl` or another enhancer-specific nested project-state schema.

Keep the previous successful active image until upload + authoritative Storyboard patch succeed.

Storyboard completion does not rewrite Director simply because a segment may reference the scene.

---

## 18. Director writeback

For local `video_upscale` / `vfi_only`, patch only:

```text
project_video_timeline.segments[n].upscaledVideoUrl
```

Do not add new Director enhancer provenance fields such as:

```text
enhancedVideoUrl
enhancementMetadata
enhancerJob
```

Detailed job/output provenance stays in `pending_upscales` and existing media/storage ownership state.

Keep the previous successful video active until the new output exists and the segment patch commits.

### 18.1 Still-image skip

Before creating a Director video enhancement job:

```text
active/current authoritative media is video -> eligible
active/current authoritative media is image -> skipped_image
```

For `skipped_image`:

```text
no video enhancer job
no pod allocation
no video enhancement charge
no video output
leave image state unchanged
```

Bulk progress must count skipped images separately from failures.

---

## 19. Private R2 TensorRT layout and trust boundary

Canonical private root:

```text
models/.engine/
```

Only these model directories are allowed:

```text
models/.engine/realesr-animevideov3/
models/.engine/realesr-general-x4v3/
models/.engine/rife-4.9/
```

No image-ESRGAN, FlashVSR or GIMM engine directories.

Compatibility directories may include:

```text
ampere-plus/              # only when an AMPERE_PLUS engine was deliberately built/validated
ampere-cc86/
ada-cc89/
blackwell-cc120/
exact/<gpu-slug>/         # only when deliberately generated
```

Avoid literal `+` in object keys.

### 19.1 Immutable filenames

Never use one mutable shared `engine.plan` filename.

Filename identity must include enough of:

```text
model/version
ONNX hash
TensorRT version
precision
shape/profile
SceneBuilder engine/build id
```

### 19.2 Temporary upload and promotion

Builder uploads first to:

```text
models/.engine/.tmp/{engineBuildJobId}/...
```

Worker verifies:

```text
expected job/pod binding
object existence
size
SHA-256
model/version
ONNX hash
TensorRT/CUDA compatibility
precision
shape/profile
self-test result
```

Only then copy/promote to immutable final key and activate in D1.

R2 upload alone never makes an engine active.

### 19.3 D1 is authoritative

Runtime lookup:

```text
D1 active compatible engine metadata
-> exact immutable R2 key
-> download
-> checksum
-> deserialize
-> inference self-test/use
```

Do not scan R2 folders to guess the active engine. Optional sidecar JSON is diagnostics/disaster-recovery only.

### 19.4 Trusted engines only

Never accept arbitrary user-supplied `.engine` files or browser-supplied ONNX/build commands.

Only engines built through the authenticated SceneBuilder engine-build path from trusted baked ONNX/model artifacts may become active.

---

## 20. TensorRT portability/selection

Support these compatibility targets in admin/registry logic:

```text
Ampere+ portable        # TensorRT AMPERE_PLUS, only if benchmarked/validated
Ampere CC 8.6           # SAME_COMPUTE_CAPABILITY
Ada CC 8.9              # SAME_COMPUTE_CAPABILITY
Blackwell CC 12.0       # SAME_COMPUTE_CAPABILITY
Exact GPU               # device-specific, deliberately generated
```

Normal runtime lookup preference:

```text
1. exact-GPU active engine when one exists and was deliberately preferred
2. same-compute-capability active engine
3. validated Ampere+ portable engine
4. native full-frame PyTorch CUDA fallback
```

Do not automatically create every compatibility variant. Start with required family sets, benchmark portable/specific alternatives, and multiply artifacts only when evidence justifies it.

Engine key includes at least:

```text
model family/version/checkpoint hash
ONNX SHA-256
precision
TensorRT version
CUDA runtime class
compatibility mode
compute capability target
shape/profile
builder flags/plugins/ABI
```

---

## 21. Engine-build lifecycle

Admin action `Generate engine` creates:

```text
pending_upscales.job_type = engine_build
status = queued
```

Flow:

```text
Worker validates ALLOWED_EMAILS + server allowlists
-> resolves trusted model/ONNX hash/profile/precision
-> resolves provider and target compute capability
-> reuses compatible idle FAST pod or provisions builder pod
-> pod verifies GPU/CUDA/TensorRT
-> builds from baked ONNX
-> deserialize/self-test
-> correctness inference
-> optional benchmark against native PyTorch CUDA
-> SHA-256
-> upload temporary private R2 object + optional timing cache
-> authenticated callback
-> Worker verifies job/pod/object/metadata/checksum
-> promote immutable final R2 object
-> D1 transaction activates new engine
-> previous matching active engine becomes inactive only after new engine is ready
```

A failed replacement never disables the old active engine.

The engine-builder uses the normal FAST image; do not maintain a separate builder image. After building, a compatible pod may serve normal FAST jobs until idle timeout.

Build states:

```text
queued
waiting_for_pod
selecting_gpu
provisioning
booting
building
self_test
benchmarking
uploading
verifying
ready
failed
cancelled
```

### 21.1 Builder GPU selection

A SAME_COMPUTE_CAPABILITY engine is built on a validated GPU of that compute capability, not merely a marketing-adjacent GPU.

Example:

```text
Ada CC 8.9 -> build on a validated CC 8.9 GPU such as RTX 4090/L40S/RTX 6000 Ada
Blackwell CC 12.0 -> build on validated CC 12.0 GPU
Ampere CC 8.6 -> build on validated CC 8.6 GPU
```

If an engine build OOMs, retry a higher-memory GPU within the same requested compatibility class before changing compatibility target.

---

## 22. Provider scheduling

Use RunPod and Novita adapters with provider-neutral enhancer jobs.

RunPod creation mirrors the useful H3 pattern:

```text
name
imageName
gpuTypeIds
gpuTypePriority=custom
gpuCount=1
computeType=GPU
compatible CUDA/driver requirement
container disk
env
ports
interruptible=false
locked=false
```

Novita includes:

```text
name
productId
gpuNum
rootfsSize
imageUrl
ports
envs
minCudaVersion / current equivalent
billing/network fields required by API
```

Provider env contains enhancer-specific worker token/service kind, R2 access, idle timeout, debug/runtime flags and ports.

`Auto` may cross providers. If admin explicitly forces Novita or RunPod, do not silently cross to the other provider.

GPU allocation is a control-plane/runtime concern, not a Docker rebuild concern.

---

## 23. Runtime concurrency and OOM policy

### 23.1 Admin execution controls

Expose:

```text
Execution
  Sequential
  Parallel

Parallelism
  x1
  x2
  x3
  x4
  Custom
```

Initial custom safety range:

```text
1-8
```

Changing xN does not rebuild Docker or TensorRT engines.

### 23.2 Storyboard image batching

Exact-shape V1 eligibility:

```text
same image ESRGAN model
same backend
same target
same exact input width
same exact input height
```

Same aspect ratio alone is not enough.

Default automatic behavior:

```text
exact same W x H -> up to x4
mixed W x H      -> sequential on the same warm pod
```

Do not resize/pad images solely to manufacture a matching batch.

If fewer than four matching images are available, process the available group rather than waiting indefinitely.

Persist image shape metadata in D1 so the Worker, not only the frontend, is authoritative for batching.

### 23.3 One video job per GPU

For local video enhancement:

```text
one GPU = one active video job
```

Queued videos on a warm pod run one after another.

For video, `xN` means **internal work inside the one active clip**, never N simultaneous videos.

Video ESRGAN:

```text
xN = up to N compatible decoded frames/tasks in flight
```

RIFE:

```text
xN = up to N interpolation pair/timestep tasks in flight
```

FlashVSR:

```text
default x1 temporal/chunk pipeline
x2+ only after exact GPU/resolution validation
```

GIMM:

```text
bounded validated task/window concurrency only; default conservatively until measured
```

### 23.4 OOM backoff

Default:

```text
Auto-backoff on CUDA OOM = ON
```

For requested concurrency:

```text
x4 -> x3 -> x2 -> x1
```

For custom, decrement to a safe lower value.

On OOM:

1. stop new submissions;
2. synchronize affected CUDA work;
3. release failed contexts/buffers;
4. clear reclaimable framework caches where appropriate;
5. retry at lower concurrency;
6. if full-frame sequential still OOMs, requeue to the next allowed higher-VRAM GPU class.

Do not tile ESRGAN/RIFE and do not fall back to CPU neural inference.

Persist unsafe GPU/model/resolution/backend/concurrency combinations so the scheduler does not repeat known-bad settings.

---

## 24. Pod API and processing stages

Enhancer runtime should expose at least:

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
normalizing
upscaling
interpolating
encoding
uploading
verifying
done
failed
cancelled
```

`/capabilities` reports actual GPU/runtime identity and only passed model/backend capabilities.

---

## 25. Security/authentication

Follow H3's security style while keeping enhancer-specific state/domain separation.

Requirements:

- authenticate job submission/cancel/admin routes;
- separate enhancer callback endpoint, e.g. `/api/projects/v2/enhancer/pod/events`;
- enhancer HMAC derivation message domain uses `enhancer:${workerId}`;
- master secret may come from `ENHANCER_POD_AUTH_MASTER_SECRET || H3_POD_AUTH_MASTER_SECRET`, but the enhancer derivation domain stays distinct;
- signed/job-scoped callback token;
- timestamp + nonce replay protection through `enhancer_event_nonces`;
- timing-safe MAC/token comparison;
- callback bound to expected job and expected worker/pod;
- do not trust callback-supplied arbitrary endpoint/IP data;
- trusted signed R2 URLs or scoped R2 credentials;
- server-side allowlists for engine model/profile/provider/GPU/precision/encoder ranges;
- browser cannot supply arbitrary ONNX URLs or shell commands;
- audit engine-build/admin metadata including requesting email, provider, GPU, model/profile and result.

### 25.1 Admin authorization source

Use exactly the existing SceneBuilder `ALLOWED_EMAILS` source from `wrangler.toml` for enhancer admin visibility **and** server-side authorization.

Do not add a separate enhancer-admin allowlist.

---

## 26. Lifecycle behavior

Enhancer lifecycle includes:

```text
priority dispatch
idle worker reuse
persistent admin idle timeout
provision timeout
job timeout
heartbeat/progress
structured error/debug state
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

Timeouts delete/terminate abandoned provider instances through enhancer-owned delete locks, never H3 locks.

Cancellation is cooperative and checked before/between major stages and frame/chunk work. Terminate FFmpeg/model subprocess work safely where applicable.

---

## 27. Storyboard Upscale UI/backend

Preserve the existing Storyboard user flow:

```text
Upscale Image
Upscale All
Anime / Real
2K / 4K
batch progress
```

Storyboard mapping:

```text
Anime -> RealESRGAN_x4plus_anime_6B
Real  -> RealESRGAN_x4plus
```

`Upscale All` local batch uses the enhancer FAST pool only and never Replicate.

### 27.1 Batch scope

For `Upscale All`, use confirmation scope where appropriate:

```text
Upscale remaining
Upscale all including already-upscaled
```

Default: `Upscale remaining`.

If including already-upscaled is chosen, start from committed/base pre-upscale image, never recursively from the current upscaled output. Keep the previous successful upscale active until replacement succeeds.

### 27.2 Storyboard admin surface

Allowed-email users get a collapsed enhancer admin section near the existing Storyboard upscale controls.

Show only relevant controls:

```text
provider Auto/Novita/RunPod
GPU family/exact target
the current FAST runtime/capabilities
Sequential/Parallel x1/x2/x3/x4/Custom
OOM auto-backoff
idle timeout
pod/telemetry/errors/debug
```

Do **not** show TensorRT engine build controls for Storyboard image ESRGAN because image ESRGAN has no `.engine` by policy.

---

## 28. Video Generation Timeline / Director UI

Reuse the existing Video Generation Timeline UI. Do not create a separate normal-user enhancer page.

Extend the existing Video/Image track and Controls -> Upscale surfaces.

### 28.1 Existing batch actions

Preserve/extend:

```text
Upscale All
Upscale Selected
per-unit Controls -> Upscale
```

Clicking `Upscale All` opens a confirmation modal with counts such as:

```text
Batch Upscale

Eligible videos: N
Already upscaled: N
Remaining: N
Skipped still images: N

○ Upscale remaining
○ Upscale all including already-upscaled
```

Default: `Upscale remaining`.

`Upscale Selected` uses the same semantics limited to selected logical units.

Force-reprocessing starts from the authoritative pre-enhancement source, never the previous enhanced master. Keep the old enhanced master active until replacement succeeds.

### 28.2 User upscaler dropdown

Keep one normal-user `Upscaler` selector; do not split the UI into separate Local/Topaz pages.

Options:

```text
Topaz · Gaia 2
Topaz · Proteus
Real-ESRGAN · Anime
Real-ESRGAN · Real
FlashVSR v1.1
```

Output target:

```text
1080p
1440p
2160p / 4K
```

### 28.3 User VFI dropdown

Use an explicit VFI selector rather than a boolean `interpolateTo60Fps`.

For local upscalers:

```text
None
RIFE 4.9
GIMM-VFI-F
```

For Topaz upscalers, show only VFI options actually supported by the Topaz backend, e.g.:

```text
None
Topaz · Chronos
Topaz · Apollo   # only where backend/product currently supports it
```

Changing upscaler immediately revalidates the VFI selection. Do not display incompatible disabled choices; reset invalid selection to `None`.

### 28.4 Target FPS

When local/Topaz VFI is enabled, show explicit target FPS supported by product policy, including:

```text
30
48
60
```

The backend supports exact arbitrary timestamp scheduling for RIFE, so mappings such as 30 -> 48 are valid and do not create another engine.

When VFI is None, preserve source/master FPS and hide VFI-only controls.

### 28.5 Director playback-speed option

When VFI is enabled, expose:

```text
☑ Apply Director playback speed · <current speed>x
```

Default checked as in the older design unless existing product behavior dictates otherwise at implementation time.

Checked -> VFI produces timing appropriate to the current Director speed/range.

Unchecked -> VFI runs at normal source/master timing; Director speed remains a later TimelineRender operation.

Do not double-apply speed.

### 28.6 Bulk progress

Distinguish at least:

```text
queued
processing
completed
failed
skipped_image
cancelled
```

Still images are `skipped_image`, not failed.

---

## 29. Video Enhancer admin UI

Use the existing `ALLOWED_EMAILS` authorization source and follow the existing H3 admin-control pattern inside Video Generation Timeline / Controls -> Upscale.

Normal users do not see these controls; every API action still checks authorization server-side.

### 29.1 Pod/runtime controls

Expose:

```text
Idle timeout seconds
Priority
Provider: Auto / Novita / RunPod
GPU family / exact GPU
Execution: Sequential / Parallel
Parallelism: x1 / x2 / x3 / x4 / Custom
Auto OOM backoff
active/idle pods
current job
GPU/VRAM
runtime image tag + digest
git/build revision
CUDA
PyTorch
CuPy
TensorRT
loaded model
loaded engine + checksum
telemetry
last heartbeat
last structured error/debug log tail
Stop/Delete pod
Manual dispatch
```

### 29.2 TensorRT engine builder UI

`Generate engine` is available **only** for:

```text
Video ESRGAN · Anime (realesr-animevideov3)
Video ESRGAN · Real  (realesr-general-x4v3)
RIFE 4.9
```

Never show image ESRGAN, FlashVSR or GIMM as TensorRT build targets.

Form:

```text
Model family
Model/checkpoint/ONNX hash (read-only/trusted manifest)
Precision: FP16 default; no INT8 in V1
Compatibility target:
  Ampere+ portable
  Ampere CC 8.6
  Ada CC 8.9
  Blackwell CC 12.0
  Exact GPU
Provider:
  Auto
  Novita
  RunPod
Shape/profile:
  six ESRGAN static shapes, or
  RIFE 1080/1440/2160 class
```

Actions:

```text
Generate engine
Validate engine
Benchmark engine
Force rebuild
Deactivate engine
Delete cached engine
Copy R2 key
View build logs
```

Inventory shows:

```text
model
shape/profile
precision
compatibility target
GPU family/exact GPU
TensorRT/CUDA
provider/build GPU
file size
build duration
benchmark FPS/latency when measured
created/validated timestamps
status: active/stale/invalid/failed
```

### 29.3 Encoder admin controls

Expose the H.265 encoder controls from Section 15.4:

```text
NVENC + CQ (default 16)
x265 + CRF (default 15)
```

### 29.4 Benchmark admin controls

Allow A/B measurements for:

```text
TensorRT vs PyTorch CUDA
provider A vs B
GPU family A vs B
same-CC vs Ampere+ portable vs exact-GPU when available
NVENC vs x265
```

Metrics:

```text
inference-only FPS
end-to-end FPS
decode FPS/time
encode FPS/time
peak VRAM
CPU/system RAM
wall-clock time
output bytes
R2 transfer time
optional quality metrics where meaningful
```

Persist compact benchmark metadata in D1 with the relevant engine/runtime build.

---

## 30. Pricing/billing contract

Do not silently treat FAST, QUALITY and external premium compute as equal-cost operations.

Preserve existing Storyboard image billing, including current 2K/4K product pricing, unless separately changed.

Video pricing must remain explicit by product/backend and use existing billing/team-payer/refund conventions rather than a new parallel wallet system.

Rules:

- no charge for `skipped_image`;
- no duplicate charge from retries/idempotent callbacks;
- failure/cancel refunds follow the existing SceneBuilder product billing policy;
- Topaz uses its existing premium/external pricing path;
- local FAST/QUALITY prices must be configured/approved before production rather than invented inside the GPU runtime;
- never silently substitute a cheaper/different model for a paid QUALITY/Premium request unless product policy explicitly permits it.

---

## 31. Observability and structured failures

### 31.1 Telemetry

Pod `/telemetry` should expose at least:

```text
provider
provider instance id
canonical GPU name
compute capability
driver
CUDA
PyTorch
CuPy
TensorRT
VRAM total/used/free
GPU utilization
memory-controller utilization
temperature
power where available
NVENC/NVDEC utilization
CPU utilization
system RAM
active model/backend
engine key
execution concurrency
input/output dimensions
current stage
frames/chunks completed/total
model FPS
end-to-end FPS
queue depth
elapsed time
ETA when trustworthy
```

Do not write every one-second NVML sample to D1. Persist compact snapshots on start, stage transitions, restrained active intervals, completion/failure and peak/summary metrics.

### 31.2 Per-stage timings

Record enough to diagnose CPU/IO bottlenecks:

```text
ffprobe/decode
normalization/crop
model inference
VFI
encode
upload/download
R2 transfer
TRT cache hit/miss
peak VRAM/RAM
```

### 31.3 Structured error codes

Use machine-readable error codes, including at least:

```text
CUDA_UNAVAILABLE
CUDA_OOM
GPU_CAPABILITY_MISMATCH
TRT_ENGINE_NOT_FOUND
TRT_DESERIALIZE_FAILED
TRT_BUILD_FAILED
TRT_SELF_TEST_FAILED
MODEL_LOAD_FAILED
RIFE_RUNTIME_FAILED
GIMM_RUNTIME_FAILED
FLASHVSR_SELF_TEST_FAILED
FFMPEG_PROBE_FAILED
FFMPEG_DECODE_FAILED
NVENC_ENCODE_FAILED
X265_ENCODE_FAILED
R2_INPUT_FAILED
R2_OUTPUT_FAILED
PROVIDER_CAPACITY_EXHAUSTED
POD_HEALTH_TIMEOUT
POD_LOST
CANCELLED
UNKNOWN
```

Error payload includes relevant model/backend/engine/GPU/VRAM/dimensions/concurrency/CUDA/TRT/driver/retry count and a bounded recent log tail.

---

## 32. Benchmark/qualification suite

Use representative SceneBuilder clips/images, including:

```text
realistic faces
anime/stylized faces
hair/fine detail
text/signage
foliage
sky/gradients
camera pans
fast subject motion
occlusion/reveal
portrait 9:16
landscape 16:9
H3-origin .4/.6/.7/.9 source rasters
slightly off-ratio provider output such as 1344x768
```

For each applicable model/backend record FPS, wall time, VRAM, CPU/RAM, decode/pre/post/inference/encode and R2 transfer.

Before promoting a family-specific engine/runtime combination, compare TensorRT output against the trusted native PyTorch reference for gross correctness/parity. This is a validation gate, not a requirement to invent another user-visible quality mode.

---

## 33. Cold start and Docker/engine separation

Docker build and TensorRT engine generation are independent operations.

Docker rebuilds are for:

```text
code changes
model/checkpoint changes
ONNX source changes
runtime dependency changes
CUDA/PyTorch/CuPy/TensorRT changes
```

Docker rebuild is **not** required for:

```text
provider priority changes
GPU allowlist changes
x1/x2/x3/x4/custom changes
OOM policy changes
engine generation/activation/deletion
encoder CQ/CRF changes
telemetry sampling changes
```

Cold-start goals:

- model weights already inside the image;
- only selected `.engine` artifacts may need R2 download;
- stable heavy layers survive application-server changes;
- record compressed size, unpacked size, cold-pull time and boot-to-ready after real builds rather than treating old estimates as facts.

---

## 34. Engine prewarm/release strategy

Before a new runtime version is considered production-ready for a TensorRT compatibility set:

1. launch a representative compatible GPU;
2. build the required 15-engine set when that set is part of the release target;
3. self-test each engine;
4. run correctness/golden fixtures;
5. upload immutable engines/timing caches to R2;
6. write/activate D1 metadata only after verification;
7. run real GPU runtime qualification;
8. release/promote the image/runtime combination.

A normal production pod should prefer cache hits and should not repeatedly rebuild known engines.

---

## 35. Implementation phases

### Phase 1 - control plane + native GPU baseline

- preserve H3 lifecycle boundaries;
- enhancer D1 migrations/tables;
- provider adapters/lifecycle;
- FAST native image/video ESRGAN;
- RIFE 4.9 native CUDA;
- GIMM integration/testing subject to license;
- FFmpeg/NVDEC/NVENC;
- R2 media flow/writeback;
- security/callbacks;
- Storyboard Upscale All local routing.

### Phase 2 - video normalization/timing

- ffprobe metadata;
- canonical H3-origin raster mapping inside enhancer only;
- small aspect crop + Lanczos normalization;
- non-canonical PyTorch fallback;
- trim/range/playback-speed VFI semantics;
- RIFE arbitrary timestamps.

### Phase 3 - TensorRT

- trusted ONNX exports for the two video ESRGAN models + RIFE 4.9;
- 12 static video ESRGAN engines per compatibility set;
- 3 RIFE resolution-class engines per compatibility set;
- immutable R2 engine tree;
- `pending_upscales` engine registry/history;
- builder/admin flows;
- family-specific/portable benchmark logic;
- native GPU fallback.

### Phase 4 - QUALITY

- FlashVSR validated runtime;
- current CUDA/custom-kernel qualification;
- temporal/chunk processing;
- RIFE/GIMM same-pod compatibility;
- 4090-class-or-better initial QUALITY routing with capability checks.

### Phase 5 - UI/admin

- Video Generation Timeline model/VFI/FPS/speed UI;
- Upscale All/Selected modal semantics;
- Storyboard local Upscale All/admin controls;
- TensorRT builder/inventory;
- encoder controls;
- provider/GPU/concurrency controls;
- benchmark/telemetry/error UI.

### Phase 6 - optimization

- learned per-GPU concurrency defaults;
- optional validated Ampere+ portable engines;
- exact-GPU engines only where benchmark benefit justifies them;
- additional static video ESRGAN shapes only if production usage warrants them;
- performance tuning without changing product-state schema.

---

## 36. Source/license policy

Current broad policy:

```text
Real-ESRGAN -> use permissive upstream and pinned dependencies
RIFE        -> use permissive upstream; build our own TensorRT path rather than copying non-commercial wrappers
FlashVSR    -> verify/pin upstream and dependency licenses before production
GIMM-VFI    -> production blocked until commercial permission/license is confirmed
Topaz       -> external licensed/provider backend; do not redistribute assets locally
```

Community wrappers/Reddit reports are implementation/benchmark signals, not permission to copy incompatible-licensed code.

---

## 37. Items still open / not silently guessed

These are intentionally **not** hard-locked until resolved by qualification/product decision:

1. **Exact CUDA 13 minor + exact PyTorch 2.x pin.** One system CUDA line per Docker image is locked; the precise minor/version pair must pass FAST + QUALITY dependency/model tests.
2. **GIMM commercial production permission.** Technical integration can proceed only within license terms; commercial production stays blocked until permission/license is confirmed.
3. **Scene-cut-aware VFI.** Older plans proposed detecting hard cuts so interpolation does not morph across a cut. This is sensible but remains an explicit quality/performance decision until tested with SceneBuilder clips.
4. **Duplicate/static-frame VFI skipping.** Optional optimization only; do not implement in a way that changes exact timestamps/duration.
5. **Local FAST/QUALITY video price amounts.** Billing mechanics are locked; exact local video prices are a product configuration decision.
6. **Whether an AMPERE_PLUS portable TensorRT set is worth keeping.** Same-CC family sets are the safe baseline; build portable/exact-GPU sets only after benchmark evidence.
7. **Exact FlashVSR CUDA/custom-kernel compatibility matrix.** QUALITY eligibility is determined by real self-test/benchmark rather than assumed from CUDA architecture support alone.
8. **Future repository rename.** Enhancer remains under the current H3 repo now; broader repo rename is a separate later migration.

---

## 38. Final canonical capability matrix

```text
FAST
  one CUDA 13.x userspace line
  PyTorch 2.x CUDA-13 build selected by qualification
  CuPy cupy-cuda13x
  TensorRT 10.16.1
  FFmpeg NVDEC/NVENC

  IMAGE UPSCALE
    RealESRGAN_x4plus_anime_6B -> native full-frame GPU PyTorch only
    RealESRGAN_x4plus          -> native full-frame GPU PyTorch only

  VIDEO UPSCALE
    realesr-animevideov3       -> TensorRT engine preferred; full-frame GPU PyTorch fallback
    realesr-general-x4v3       -> TensorRT engine preferred; full-frame GPU PyTorch fallback

  VFI
    RIFE 4.9                   -> TensorRT engine preferred; full-frame GPU PyTorch fallback
    GIMM-VFI-F                 -> native PyTorch + CuPy only, license-gated

QUALITY
  one CUDA 13.x userspace line
  PyTorch 2.x CUDA-13 build selected by qualification
  CuPy cupy-cuda13x
  TensorRT 10.16.1 for RIFE execution
  FFmpeg NVDEC/NVENC

  VIDEO UPSCALE
    FlashVSR v1.1              -> native GPU only

  VFI
    RIFE 4.9                   -> TensorRT engine preferred; GPU PyTorch fallback
    GIMM-VFI-F                 -> native PyTorch + CuPy only, license-gated

EXTERNAL PREMIUM
  Topaz Gaia 2 / Proteus
  Topaz Chronos / Apollo where supported
  external backend only; no local TensorRT engine registry
```

TensorRT `.engine` files exist only for:

```text
realesr-animevideov3
realesr-general-x4v3
RIFE 4.9
```

Engine count per compatibility target:

```text
6 Anime video ESRGAN
6 Real video ESRGAN
3 RIFE resolution-class engines containing portrait+landscape profiles
---------------------------------------------------------------
15 engine files per compatibility set
```

No image ESRGAN TensorRT. No FlashVSR TensorRT. No GIMM TensorRT. No neural CPU fallback. No enhancer rows in H3 lifecycle tables.
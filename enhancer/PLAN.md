# SceneBuilder Enhancer Implementation Plan

Status: **canonical implementation plan**

Branch: `feat/enhancer-gpu-runtime`

This file is intentionally separate from the mature H3 lifecycle design. Enhancer code may live in the same GPU-runtime repository, but enhancer lifecycle/state remains physically separate from H3. When older enhancer drafts or earlier revisions conflict with this file, **this file wins**.

Current hard runtime lock:

```text
System CUDA userspace/toolkit: CUDA 13.0 Update 2 (13.0.2)
PyTorch:                      2.13.0 / cu130 build family
CuPy:                         cupy-cuda13x
TensorRT:                     10.14.1.48
Neural CPU fallback:          forbidden
```

Provider scheduling requests CUDA **13.0**, not CUDA 13.1. Boot-time CUDA/CuPy/model/TensorRT smoke tests are authoritative.

---

## 1. Product goal

Build a SceneBuilder image/video enhancer using RunPod and Novita GPU pods with the same operational discipline as H3 while preserving H3 implementation/state.

Supported product paths:

- Storyboard single-image upscale stays on the existing legacy path unless deliberately changed later.
- Storyboard **Upscale All** uses local enhancer pods only; zero Replicate.
- Video Generation Timeline / Director local video upscale.
- FAST and QUALITY local enhancement.
- Local VFI using RIFE 4.9 or GIMM-VFI-F.
- Existing Topaz premium/external upscale/VFI choices stay behind their existing external backend.
- Video output targets: 1080p, 1440p/2K, 2160p/4K.
- Local VFI target FPS choices: 30, 48, 60 only.
- Source FPS may be arbitrary or VFR.
- H3-style priority, retries, heartbeat, debug/error state, idle reuse, configurable idle timeout, provision/job timeout, provider fallback, delete locks, verification/backoff, stale-allocation cleanup, cancel and orphan cleanup.

Spatial enhancement should create a reusable enhanced master. Trim/speed/FPS-only changes should rerun only timing/VFI where possible, not ESRGAN/FlashVSR again.

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
- H3 code never queries enhancer lifecycle tables.
- Enhancer never treats H3 workers as its pool.
- Enhancer has its own callback endpoint, HMAC derivation domain, replay state, delete locks and reaper state.
- Mature H3 runtime/lifecycle source behavior is not modified to make enhancer work.
- Stateless provider/R2 helper patterns may be reused only where they do not share lifecycle state.

### 2.2 Product D1 remains authoritative

Do not create a second project-media table.

`project_video_timeline` remains the Director product state and continues to store revisioned/batched `data_json`. Minimal additive Director segment fields needed for active asset selection, Undo Upscale and timing correctness are approved in Section 19. Detailed execution/provenance remains in `pending_upscales`.

Storyboard product state remains in `projects_timeline`.

### 2.3 Product R2 layout remains authoritative

Permanent semantic destinations:

```text
Storyboard image upscale:
projects/{projectId}/images/scenes/upscaled/

Director video upscale:
projects/{projectId}/video/upscaled/
```

No product-media R2 migration is part of enhancer work.

### 2.4 Replicate boundary

- `POST /api/upscale-batch` / Storyboard **Upscale All** -> local enhancer FAST pods only.
- New local jobs use enhancer-owned states such as `waiting_for_pod`, never legacy Replicate-claimable state.
- Existing single-image `/api/upscale` stays legacy Replicate unless deliberately changed later.

### 2.5 FILM is deleted

FILM is not installed, exposed, routed, benchmarked, documented as supported, or stored anywhere in enhancer.

---

## 3. Final model/runtime matrix

Two local Docker image families:

```text
scenebuilder-enhancer-fast
scenebuilder-enhancer-quality
```

### 3.1 FAST

```text
CUDA 13.0 Update 2
PyTorch 2.13.0 cu130
CuPy cupy-cuda13x
TensorRT 10.14.1.48 builder + runtime
FFmpeg with NVDEC/NVENC where available
NVML telemetry

Storyboard IMAGE:
  Anime -> RealESRGAN_x4plus_anime_6B
  Real  -> RealESRGAN_x4plus

Director VIDEO:
  Anime -> realesr-animevideov3
  Real  -> realesr-general-x4v3

VFI:
  RIFE 4.9
  GIMM-VFI-F
```

### 3.2 QUALITY

```text
CUDA 13.0 Update 2
PyTorch 2.13.0 cu130
CuPy cupy-cuda13x
TensorRT 10.14.1.48 runtime for compatible RIFE engines
FFmpeg with NVDEC/NVENC where available
NVML telemetry

Video upscale:
  FlashVSR v1.1 / qualified FlashVSR+ implementation path

VFI:
  RIFE 4.9
  GIMM-VFI-F
```

### 3.3 Topaz remains external

Keep existing premium/external choices:

```text
Topaz Gaia 2
Topaz Proteus
Topaz Chronos where supported
Topaz Apollo where supported
```

Do not bake/redistribute Topaz assets in local enhancer Docker images or local TensorRT storage.

---

## 4. CUDA / PyTorch / CuPy / TensorRT lock

Both FAST and QUALITY use one CUDA userspace line:

```text
CUDA Toolkit/userspace: 13.0 Update 2 (13.0.2)
PyTorch:                 2.13.0 cu130
CuPy:                    cupy-cuda13x
TensorRT:                10.14.1.48
```

Do not install CUDA 11/12/13.1 side-by-side. Do not allow third-party installers to add another CuPy CUDA family.

TensorRT 10.14.1.48 is the V1 pin because it is an NVIDIA-tested TensorRT 10.x release for CUDA 13.0 Update 2. Do not move to TensorRT 11.x until our exporters/runtime are intentionally migrated and parity-tested.

Provider scheduling requests CUDA 13.0-class hosts where the provider exposes a CUDA selector. Boot qualification checks actual driver/runtime compatibility.

Mandatory boot checks:

```text
nvidia-smi / driver
CUDA allocation + real CUDA op
PyTorch CUDA availability/version
CuPy CUDA kernel
TensorRT load/build smoke where applicable
model-specific smoke tests
NVENC smoke where configured
```

Incompatible hosts never become ready; delete/reprovision. No CPU workaround.

---

## 5. GPU fleet policy

Supported families:

```text
Ampere    -> primarily sm_86
Ada       -> sm_89
Blackwell -> sm_120 RTX/workstation
```

Global product cap:

```text
physical VRAM <= 48 GB, including 48 GB
```

Representative candidates, subject to live provider inventory and qualification:

```text
Ampere: RTX 3090, RTX A4000/A4500/A5000, A40 48GB, RTX A6000 48GB
Ada:    RTX 4090, L4, L40/L40S 48GB, RTX 6000 Ada 48GB
Blackwell <=48GB: RTX 5090 and qualified workstation variants
```

Exclude A100/A800/H100/H200/B200 and other high-cost product-outlier classes even if technically usable.

QUALITY/FlashVSR starts at an approved RTX-4090-class floor or validated equal/higher allowed GPU.

FAST + GIMM-VFI-F has an additional hard floor:

```text
physical VRAM >= 24 GB
```

RunPod/Novita inventory filtering and boot-time NVML verification both enforce it. Forced admin selection below 24 GB is rejected; do not silently substitute RIFE.

---

## 6. Model precision policy

Precision here means **neural inference working precision**, not source media bit depth or output codec bit depth.

### 6.1 Locked V1 precision matrix

| Model | V1 primary precision | TensorRT | Allowed fallback | Notes |
|---|---|---|---|---|
| `RealESRGAN_x4plus_anime_6B` | FP16 PyTorch CUDA | none | GPU FP32 only for emergency parity/debug if a specific op proves unsafe | Official Real-ESRGAN inference is FP16 by default. |
| `RealESRGAN_x4plus` | FP16 PyTorch CUDA | none | GPU FP32 only for emergency parity/debug | Same FP16 policy as official Real-ESRGAN. |
| `realesr-animevideov3` | FP16 | FP16 preferred | native PyTorch CUDA FP16 | No BF16/FP8 engine. |
| `realesr-general-x4v3` | FP16 | FP16 preferred | native PyTorch CUDA FP16 | Fixed balanced denoise preset. |
| `RIFE 4.9` | FP16 | FP16 preferred | native PyTorch CUDA FP16; GPU FP32 only for a resolution/profile that fails exact FP16 parity qualification | Dynamic runtime timestep. No BF16/FP8 engine. |
| `GIMM-VFI-F` | **FP32 PyTorch/CuPy** | none | none in V1 | Do not silently AMP/cast the full FlowFormer/GIMM path until our own parity suite approves it. |
| `FlashVSR v1.1` | **BF16** | none | FP16 only if a specific GPU/kernel path is validated for equal quality/stability | Official v1.1 full inference uses BF16. |

### 6.2 FP8 is not used in V1

No model uses FP8 in production V1:

```text
4 Real-ESRGAN models -> no FP8
RIFE                  -> no FP8
GIMM-F                -> no FP8
FlashVSR              -> no FP8
```

Do not add FP8 TensorRT builds/calibration or framework FP8 autocast merely because Blackwell/Ada hardware supports it. It would create another quality/quantization validation problem with no approved product benefit.

### 6.3 RIFE high-resolution safety

RIFE has an official FP16 inference path, but historical RIFE builds have shown high-resolution FP16 artifact reports. Therefore the exact RIFE 4.9 checkpoint/export must pass golden parity at each activated spatial class:

```text
1080
1440
2160/4K
portrait mirrors
non-0.5 timesteps
```

If a specific 2160/profile FP16 path fails parity, do not activate that TensorRT engine/profile. Use native **GPU FP32** for that class until fixed. This is still GPU-only; it is not CPU fallback.

### 6.4 GIMM precision is intentionally conservative

Official GIMM-VFI-F uses FlowFormer plus PyTorch/CuPy/custom warping pieces and does not publish BF16/FP16 as its canonical inference contract. V1 therefore uses FP32 for correctness. The 24 GB minimum exists partly to make this conservative choice practical.

A future BF16 or mixed-precision GIMM path requires explicit golden tests for:

```text
flow stability
occlusion edges
hard motion
2K/4K DS_SCALE behavior
arbitrary timestep output
CuPy/custom op compatibility
```

### 6.5 FlashVSR BF16

FlashVSR v1.1 QUALITY uses BF16 by default. Official v1.1 full inference initializes the model manager and input tensors in `torch.bfloat16`. Our BF16 path must still qualify its sparse-attention implementation on sm86/sm89/sm120.

FP16 may exist only as a measured fallback for a GPU/kernel combination that cannot run the approved BF16 path while retaining parity. FP32 is debug/reference only because of VRAM/throughput cost.

### 6.6 Precision observability

Persist/advertise actual execution dtype:

```text
requested_precision
actual_precision
engine_precision
model/version
GPU/compute capability
peak VRAM
```

TensorRT engine identity always includes precision. V1 TensorRT artifacts are FP16.

---

## 7. GPU-only inference invariant

Neural inference may not silently use CPU.

```text
torch.cuda.is_available() == false -> worker unhealthy
model params/buffers unexpectedly on CPU -> fail/requeue
input tensors on CPU at neural boundary -> fail/requeue
framework/model offload to CPU -> capability/job failure
```

Explicitly own `CUDA_VISIBLE_DEVICES`, `cuda:0`, model/input/output device and synchronization around diagnostics/benchmarks.

CPU remains valid for HTTP, JSON, R2 I/O, filesystem, ffprobe, orchestration, audio, codecs and controlled pre/post work. It is not a neural fallback.

FAST startup qualification:

```text
CUDA op
CuPy kernel
Storyboard ESRGAN FP16 CUDA
video ESRGAN FP16 CUDA
available video ESRGAN TRT deserialize/inference
RIFE native FP16 + available TRT
GIMM-F FP32 CUDA/CuPy
FFmpeg/NVENC smoke
```

QUALITY startup qualification:

```text
CUDA op
CuPy kernel
FlashVSR BF16 representative inference
RIFE native/available TRT
GIMM-F FP32 CUDA/CuPy
FFmpeg/NVENC smoke
```

Advertise only capabilities that actually pass.

---

## 8. Exact TensorRT scope and topology

TensorRT `.engine` files exist for exactly:

```text
1. realesr-animevideov3
2. realesr-general-x4v3
3. RIFE 4.9
```

Never create TensorRT engines for:

```text
RealESRGAN_x4plus_anime_6B
RealESRGAN_x4plus
FlashVSR
GIMM-VFI-F
```

### 8.1 Video ESRGAN engines

Per video ESRGAN model:

```text
Landscape
854x480
1056x594
1280x720

Portrait
480x854
594x1056
720x1280

= 6 static full-frame FP16 engines/model
```

Therefore:

```text
AnimeVideo-v3   = 6
General-x4v3    = 6
ESRGAN total    = 12
```

No tiled ESRGAN TensorRT and no hidden ESRGAN tiling fallback. Missing/nonmatching engine -> native full-frame PyTorch CUDA FP16. If x1 full-frame still OOMs, move to a larger allowed GPU or fail/retry.

### 8.2 RIFE engines

Exactly three engine files per compatibility set:

```text
1080 class
  1920x1080 + 1080x1920 profiles

1440 class
  2560x1440 + 1440x2560 profiles

2160 class
  3840x2160 + 2160x3840 profiles
```

Landscape + portrait are profiles inside the same resolution-class engine.

RIFE ONNX/TensorRT must expose runtime inputs equivalent to:

```text
img0
img1
timestep
```

`timestep` must remain dynamic, never constant-folded to 0.5.

### 8.3 Engine count

Per compatibility target:

```text
12 video ESRGAN
 3 RIFE
----------------
15 engines
```

Retained baseline sets:

```text
sm86 same-CC       15
sm89 same-CC       15
sm120 same-CC      15
AMPERE_PLUS        15
----------------------
TOTAL              60
```

Exact-GPU sets add another 15 only when deliberately generated and benchmarked.

### 8.4 Engine identity excludes timing

These never create another engine:

```text
clip duration
frame count
source FPS
target FPS
24->48
24->60
16->60
20->30
35->48
35->60
2x/3x/4x interpolation ratio
runtime xN concurrency
trim/range
Director playback speed
```

Engine key is based on model/version/checkpoint/ONNX hash, TensorRT/CUDA compatibility, precision, compute compatibility, shape/profile, builder flags/plugin ABI and validation identity.

---

## 9. TensorRT portability, building and trust

Supported artifact modes:

```text
Exact GPU
SAME_COMPUTE_CAPABILITY sm86
SAME_COMPUTE_CAPABILITY sm89
SAME_COMPUTE_CAPABILITY sm120
AMPERE_PLUS portable
```

Runtime preference:

```text
1. exact-GPU validated active engine
2. same-CC validated active engine
3. validated AMPERE_PLUS engine
4. native GPU PyTorch fallback
```

The CUDA 13.0 FAST image carries TensorRT builder + runtime. Engine generation is admin/offline and may be slow; tactic profiling taking minutes is acceptable because artifacts are generated once and reused.

Build lifecycle:

```text
admin Generate engine
-> pending_upscales job_type=engine_build
-> validate ALLOWED_EMAILS + allowlists
-> resolve trusted baked ONNX/checkpoint hash
-> reuse/provision compatible FAST GPU pod
-> verify CUDA/TRT/GPU
-> build FP16
-> deserialize/self-test
-> correctness comparison vs native CUDA reference
-> benchmark when requested
-> SHA-256
-> upload private R2 temp key
-> authenticated callback
-> Worker validates expected job/pod/object/hash/metadata
-> promote immutable final key
-> D1 activation transaction
-> deactivate prior matching engine only after replacement succeeds
```

Failure never removes the previous active engine.

Before deserialize, verify trusted private R2 ownership, exact key, SHA, model/version, TRT/CUDA, compatibility/profile and expected job/worker. Never deserialize arbitrary user engine blobs.

---

## 10. TensorRT artifacts: Docker / R2 / D1

Docker owns durable sources:

```text
model/runtime code
weights/checkpoints
video-ESRGAN ONNX/export source
RIFE ONNX/export source
TRT builder/runtime
FFmpeg/NVML service
```

Generated `.engine` files are never baked into Docker.

Private R2 root:

```text
models/.engine/
```

Allowed trees only:

```text
models/.engine/realesr-animevideov3/
models/.engine/realesr-general-x4v3/
models/.engine/rife-4.9/
```

Compatibility paths include:

```text
ampere-plus/
ampere-cc86/
ada-cc89/
blackwell-cc120/
exact/<gpu-slug>/
```

Temporary build uploads:

```text
models/.engine/.tmp/{engineBuildJobId}/...
```

Use immutable filenames encoding identity. Never use one mutable shared `engine.plan`.

D1 stores metadata/control state, not engine binary bytes. `pending_upscales` remains the unified V1 job/history/engine-registry source of truth; no parallel product-media table is introduced for engines.

Runtime receives trusted engine identity + exact R2 key + expected SHA + scoped/short-lived access, downloads into disposable local storage, verifies, then deserializes. No provider network volume is required.

---

## 11. Video ESRGAN source-raster normalization

Known H3-origin videos may map to enhancer engine rasters; this does **not** put enhancer code/models into H3 pods.

Landscape mapping:

```text
H3 .4 -> 854x480  -> exact
H3 .6 -> 1056x594 -> exact
H3 .7 -> 1138x640 -> may normalize to 1056x594 when quality rule permits
H3 .9 -> 1280x720 -> exact
```

Portrait mirrors.

Never stretch. For small aspect mismatch, crop excess first and Lanczos resize to canonical raster.

Example:

```text
1344x768
-> crop 1344x756
-> Lanczos 1280x720
-> TRT
```

Noncanonical fallback:

```text
matching TRT -> TRT
safely close -> crop/resize -> TRT
otherwise -> native full-frame PyTorch CUDA FP16
```

Never downscale 1080p to 720 merely to hit a TRT engine. 1080->1080 spatial enhancement is skipped unless restoration is explicitly requested. 1080->1440/4K stays native GPU until a justified profile exists.

Neural x4 scale and final delivery resolution are separate. Exact final conform happens after neural inference.

---

## 12. FPS/VFI contract

Product target FPS values are exactly:

```text
30
48
60
```

Source rate may be arbitrary or VFR.

For CFR source FPS `S`, target `T`, playback speed `p` when timing is intentionally baked:

```text
outputTime[n]  = n / T
sourcePosition = outputTime[n] * S * p
left           = floor(sourcePosition)
right          = left + 1
alpha          = sourcePosition - left
```

Exact source timestamp -> reuse source frame. Otherwise interpolate with runtime `alpha`.

RIFE:

```text
RIFE(frame[left], frame[right], timestep=alpha)
```

GIMM-F:

```text
GIMM-F(frame[left], frame[right], timestep=alpha)
```

Same spatial RIFE engine for all FPS ratios.

If target FPS <= source sampling rate, do deterministic temporal resampling/downsampling; do not waste RIFE/GIMM compute just to reduce FPS. Normalize an unused VFI selection to None so the user is not charged for model work that did not execute.

For VFR, use real source PTS, bracket each target timestamp and compute:

```text
alpha = (targetPTS - leftPTS) / (rightPTS - leftPTS)
```

### 12.1 Scene-cut-aware VFI is mandatory

Never interpolate across a hard scene cut. Close the old scene, do not synthesize a cross-cut morph, and restart interpolation context in the next scene.

### 12.2 No duplicate/static-frame optimization

Do not drop/collapse/substitute duplicate-looking or static frames to save compute. Preserve authoritative source timestamps. Disable third-party static-skip options.

---

## 13. GIMM-VFI-F

Locked variant:

```text
GIMM-VFI-F
FlowFormer estimator
native PyTorch + CuPy CUDA
FP32 V1 precision
minimum physical VRAM 24 GB
```

Do not substitute GIMM-VFI-R or perceptual `*-P` variants. No GIMM TensorRT.

Commercial production remains license-gated until appropriate commercial permission/license is confirmed.

---

## 14. FlashVSR / QUALITY

FlashVSR is QUALITY-only and native GPU:

```text
FlashVSR v1.1 / qualified FlashVSR+ lineage
BF16 primary
no TensorRT
no CPU neural fallback
4090-class-or-better initial floor
```

Architecture ideas approved after our own validation:

```text
2x/4x neural SR scale
optional high-quality pre-resize for final delivery raster
scene detection
scene/chunk processing
restartable boundaries
GPU/output-dependent temporal chunk size
DiT tiling for VRAM control
largest validated tile / fewer tiles preferred
adjust overlap if seams appear
VAE tiling OFF by default in full-model path
RIFE/GIMM separate VFI stage
FFmpeg audio preservation
H.265/10-bit output
queue/cancel/health/progress
```

FlashVSR-specific DiT tiling is allowed; the no-tiling rule for video ESRGAN/RIFE remains.

Prefer full/non-DiT-tiled FlashVSR when validated VRAM permits. Otherwise use the largest validated DiT tile that avoids OOM/shared-memory spill. Do not use system RAM/offload as a hidden VRAM substitute.

SECourses Upscaler Pro is a reference for operational ideas, not a dependency and not proof of the exact backend repo. No ComfyUI runtime is required.

---

## 15. Streaming video, codec and audio

Persistent pipeline:

```text
authoritative input
-> ffprobe
-> decode/NVDEC where available
-> trim/range/timestamp preparation
-> aspect/canonical normalization
-> spatial upscale
-> optional scene-aware VFI
-> exact final conform
-> H.265 encode/NVENC default
-> audio handling/remux
-> R2 upload
-> authenticated completion callback
```

Never implement as thousands of JPEG/PNG frame dumps or one process/model load per frame. Keep model/engine persistent and stream/chunk with bounded buffers.

Default enhanced master:

```text
H.265 / HEVC
hevc_nvenc
Main10
10-bit 4:2:0 validated path
hvc1
progressive
SAR 1:1
exact requested raster/FPS
```

Admin encoder controls:

```text
NVENC CQ default 16, validated range
x265 CRF default 15, validated range
```

CPU x265 encoding is allowed; CPU neural inference is not.

Unchanged timing -> preserve/remux original audio. VFI alone does not change duration. Director playback-speed audio timing belongs to the timeline/render layer unless slow motion is intentionally baked as described below.

---

## 16. Runtime concurrency / OOM

One GPU = one active **video job** in V1.

`xN` means internal task concurrency inside that one job:

```text
video ESRGAN -> up to N same-size frame tasks
RIFE         -> up to N pair/timestep tasks
GIMM         -> bounded only when validated
FlashVSR     -> conservative scene/chunk concurrency; x1 initially unless validated
```

Combined spatial+VFI uses xN as a stage ceiling, never N+N simultaneously.

Admin controls:

```text
Sequential | Parallel
x1 | x2 | x3 | x4 | Custom
Custom initial range 1..8
Auto CUDA OOM backoff ON
```

OOM backoff:

```text
x4 -> x3 -> x2 -> x1
```

Persist unsafe GPU/model/backend/resolution/concurrency combinations. If x1 full-frame still fails, move to a larger eligible GPU or fail/retry. No ESRGAN/RIFE tiling and no neural CPU fallback.

Storyboard image batching groups only identical:

```text
model
backend
target
exact input width
exact input height
```

Same shape -> bounded batch/xN. Mixed shape -> sequential. Never resize/pad images just to batch.

---

## 17. Enhancer operational D1 schema

### 17.1 `enhancer_pod_workers`

Store enhancer-only lifecycle/provider/GPU/runtime ownership:

```text
worker id/service kind
provider/instance/endpoint
GPU/name/CC/VRAM/region
status/current job
runtime image/digest/build revision
CUDA/PyTorch/CuPy/TensorRT
capabilities/precision support/telemetry
heartbeat/error/debug
idle_since/idle_timeout/terminate_after
delete retry state
timestamps
```

### 17.2 `pending_upscales`

One unified operational job table for:

```text
image_upscale
video_upscale
vfi_only
engine_build
engine_validate
benchmark
```

Persist project/scene/segment/payer identity, priority, requested/actual provider/GPU/worker, model/version/backend, requested/actual precision, engine metadata, input/settings/execution/output JSON, source shape/timing metadata, status/stage/progress, compact telemetry, attempts/errors/debug and timestamps.

For video jobs persist enough provenance to prove the exact pre-upscale source and timing:

```text
source_view
source_url/object_key
source trim/range
Director speed at job creation
source duration
VFI target FPS
timing_baked
output duration
output URL/object key
model/backend/precision
```

Engine rows additionally store exact R2 key/SHA/size, ONNX/checkpoint hash, TRT/CUDA, FP16, compatibility target, CC/exact GPU, profile, builder identity, validation/benchmark and active state.

Keep separate:

```text
enhancer_pod_delete_locks
enhancer_event_nonces
enhancer_config
```

---

## 18. Storyboard writeback and UI

Completion patches existing Storyboard product fields only:

```text
scenes[n].image = new active upscaled image
storyboard.thumbnail = new thumbnail when produced
preserve storyboard.originalImage
preserve storyboard.originalThumbnail
```

Do not use prompt-enhancement `isEnhanced` as upscale provenance.

Keep previous successful image active until upload + authoritative writeback succeeds.

UI preserves:

```text
Upscale Image
Upscale All
Anime / Real
2K / 4K
batch progress
```

Batch scope:

```text
Upscale remaining
Upscale all including already-upscaled
```

Re-upscale starts from committed/base pre-upscale image, never the current upscaled image.

Storyboard admin may expose provider/GPU/priority/idle timeout/xN/OOM/telemetry/debug, but **no TensorRT image-engine controls**.

---

## 19. Director product-state audit and required media-state changes

This section is based on the current SceneBuilder2 implementation and supersedes earlier wording that Director completion should only ever write `upscaledVideoUrl`.

### 19.1 Current storage shape

`project_video_timeline` is physically:

```text
id
project_id
batch_index
data_json
revision
created_at
updated_at
```

The Director segments themselves live inside revisioned `data_json`. `packUnifiedVideoTimeline()` preserves complete segment objects rather than whitelisting individual segment fields, so the required changes below need **no ALTER TABLE / no new product table**.

Existing relevant segment state already includes fields such as:

```text
sourceVideoUrl
sourceVideoObjectKey
videoUrl
videoObjectKey
renderVideoUrl
renderVideoObjectKey
upscaledVideoUrl
upscaledVideoObjectKey
sourceTrimInMs/sourceTrimOutMs/sourceMediaDurationMs
generatedTrimInMs/generatedTrimOutMs/generatedMediaDurationMs
speed
activeVisualView
upscaleJobId/upscaleStatus/upscaleProgress
```

The generic project-media canonicalizer/reference scanner recursively processes object keys/URLs, so new R2 reference fields below are retained/protected by normal project storage accounting.

### 19.2 Current code problems that must be fixed

Current `VideoGenerationTimeline.tsx` behavior conflates base generated and upscaled media:

```text
generatedVideoForSegment()
  currently prefers:
  upscaledVideoUrl || videoUrl || renderVideoUrl
```

Current re-upscale source helper is also unsafe:

```text
getOriginalVideoForUnit()
  currently prefers:
  upscaledVideoUrl || renderVideoUrl || videoUrl || sourceVideoUrl
```

That can recursively upscale a previous upscale.

These helpers must be replaced/split before local enhancer rollout.

### 19.3 Three distinct selectable Director media variants

Expand the existing active selector to:

```ts
activeVisualView?: 'source' | 'generated' | 'upscaled';
```

Semantics:

```text
source
  -> authoritative source image/uploaded source video

generated
  -> base generated/render master BEFORE enhancer

upscaled
  -> current successful upscaledVideoUrl
```

Do not let `upscaledVideoUrl` redefine what “generated” means.

Introduce separate resolvers conceptually:

```text
sourceMediaForSegment()
baseGeneratedMediaForSegment()
upscaledMediaForSegment()
selectedDirectorMediaForSegment()
preUpscaleSourceForSegment()
```

`baseGeneratedMediaForSegment()` must use the existing canonical generated-master logic, including avoiding H.264 preview URLs when an H.265/master object is available.

### 19.4 Minimal additive Director segment fields

Keep existing:

```text
upscaledVideoUrl
upscaledVideoObjectKey
upscaleJobId/upscaleStatus/upscaleProgress
```

Add only product-state fields required to make selection/re-upscale/render deterministic:

```ts
upscaledFromView?: 'source' | 'generated';
upscaledSourceObjectKey?: string;
upscaledSourceUrl?: string;
upscaledSourceTrimInMs?: number;
upscaledSourceTrimOutMs?: number;
upscaledSourceSpeed?: number;
upscaledTimingBaked?: boolean;
upscaledTargetFps?: 30 | 48 | 60;
upscaledOutputDurationMs?: number;
```

Detailed backend/model/GPU/error provenance remains in `pending_upscales`; do not duplicate that entire payload into product JSON.

### 19.5 Original/generated media is never destroyed by upscale

After successful upscale:

```text
sourceVideoUrl/source image       -> retained
renderVideoUrl/videoUrl master    -> retained
upscaledVideoUrl                  -> added/replaced
```

Do not overwrite the source/generated pointer with the upscale.

If the upscale was created from an uploaded source video:

```text
upscaledFromView = 'source'
```

If created from generated/render master:

```text
upscaledFromView = 'generated'
```

### 19.6 Re-upscale source rule

Re-upscale **never** uses `upscaledVideoUrl` as neural input.

If current active view is `upscaled`, resolve the base from `upscaledFromView` + recorded source identity.

If current active view is `source` or `generated`, use that currently selected pre-upscale variant.

For “Upscale all including already-upscaled”, the same rule applies.

Keep current successful upscale active until replacement upload + authoritative D1 patch succeeds. Failed replacement leaves previous upscale active.

### 19.7 Source-change staleness

If a source upload is replaced or a generated master is regenerated and no longer matches `upscaledSourceObjectKey`/source identity, the old upscale is stale.

Do not recursively “update” from the stale upscale. UI should show stale state and offer re-upscale from the new base.

### 19.8 Legacy data normalization

Current legacy rows may have:

```text
activeVisualView = 'generated'
upscaledVideoUrl present
```

because old code treated upscale as generated. On new-load normalization, preserve visible behavior by mapping such rows to `activeVisualView='upscaled'` unless the row explicitly says `source`.

Infer `upscaledFromView` for legacy rows:

```text
base generated master exists -> generated
otherwise source video exists -> source
```

### 19.9 Remove/restore snapshots

`RemovedMediaSnapshot` and remove/restore logic must preserve the new upscale-selection/timing fields as well as existing upscale URL/object key. Removing media must not erase the ability to restore source/generated/upscaled state correctly.

---

## 20. Undo Upscale / Use Upscaled UI

When a successful upscale exists, Director UI must make the three assets explicit instead of hiding the difference behind one Source/Generated toggle.

Suggested compact selector/status:

```text
Source | Generated | Upscaled
```

Show only variants that exist.

When `activeVisualView='upscaled'`:

```text
Upscaled active ✓
[Undo Upscale]
[Re-upscale]
```

**Undo Upscale** means reversible selection, not destructive deletion:

```text
activeVisualView = upscaledFromView
```

Do **not** clear `upscaledVideoUrl`, do not delete R2, and do not refund. The user can later choose **Use Upscaled** to reactivate the stored successful upscale.

When an upscale exists but Source/Generated is active:

```text
[Use Upscaled]
[Re-upscale]
```

Re-upscale always starts from pre-upscale base media.

A separate destructive “Delete cached upscale” action is not part of normal Undo and, if ever added, must respect project R2 reference tracking.

Bulk `Upscale All` / `Upscale Selected` modal:

```text
Eligible videos: N
Already upscaled: N
Remaining: N
Skipped still images: N

○ Upscale remaining
○ Upscale all including already-upscaled
```

Default is remaining. Failed/cancelled jobs are eligible for remaining.

Bulk progress distinguishes:

```text
queued
processing
completed
failed
skipped_image
cancelled
```

---

## 21. Director playback speed and timing-baked VFI

The persisted Director `segment.speed` is **user intent** and must not be erased merely because an upscaled derivative has slow motion baked.

Current Director Trim & Adjust already computes timeline geometry as:

```text
Director duration = selected media duration / speed
```

So a 5-second selected range at 0.5x becomes a 10-second Director segment.

### 21.1 UI behavior for RIFE and GIMM

When local VFI is RIFE or GIMM-F:

```text
speed < 1.0
  -> show Smooth slow motion
  -> checked automatically

speed == 1.0
  -> hide/disable slow-motion control

speed > 1.0
  -> Smooth slow motion OFF
  -> Timeline Render applies speed-up normally
```

Example UI:

```text
Source: 5.0s · 24fps
Director speed: 0.5x
Target: 48fps
☑ Smooth slow motion
Output: 10.0s · 48fps
```

### 21.2 Timing-baked derivative

When checked, enhancer receives source/master, selected range, source PTS, Director speed `p`, target FPS and scene cuts.

It returns the **final slowed derivative**, not a temporary high-FPS master.

```text
outputDuration = selectedSourceDuration / p
```

Example:

```text
5 sec @ 24fps
speed 0.5x
target 48fps
-> 10 sec @ 48fps
-> ~480 output frames
-> source step = 24 * 0.5 / 48 = 0.25
```

RIFE/GIMM receives frame pairs plus runtime timesteps like 0.25/0.5/0.75. No 96fps intermediate file is required.

### 21.3 Effective speed depends on active variant

Persisted Director speed stays 0.5x in all cases. The **render DTO** decides effective playback rate from active media.

Case A — timing-baked upscale active:

```text
activeVisualView = upscaled
upscaledTimingBaked = true
upscaledSourceSpeed matches current Director speed

Timeline/Preview effective speed = 1.0
```

Because the upscale file is already 10 seconds, do not apply 0.5x again.

Case B — user clicks Undo Upscale / selects original source or generated:

```text
activeVisualView = source or generated

Timeline/Preview effective speed = persisted Director speed (e.g. 0.5)
```

The 0.5x behavior immediately comes back; the stored intent was never deleted.

Case C — upscale was created **without** Smooth slow motion:

```text
activeVisualView = upscaled
upscaledTimingBaked = false

Timeline/Preview effective speed = persisted Director speed (e.g. 0.5)
```

So a normal-duration 5s/48fps upscale still gets slowed by Timeline Render to the Director duration.

Case D — speed >1:

```text
normal-duration upscale/VFI master
Timeline/Preview speed = persisted >1 speed
```

Do not create unnecessary baked fast-motion derivatives in V1.

### 21.4 Trim/start offset for timing-baked derivative

If one segment is rendered to its own timing-baked derivative, the derivative contains the selected range itself:

```text
startTimeOffset = 0
effective speed = 1
sceneDuration = Director segment.durationMs
```

For a linked unit sharing one source and one common speed, enhancer may create one baked derivative for the union range. Then each part gets remapped baked offsets:

```text
bakedPartIn  = (sourcePartIn  - unionSourceIn) / speed
bakedPartOut = (sourcePartOut - unionSourceIn) / speed
```

and each part renders at speed 1.

If linked parts have different source assets or different speeds, do not pretend one baked derivative represents all of them; process per-part or disable full-unit baked slow-motion for that mixed unit.

### 21.5 Stale timing derivative

A timing-baked upscale is valid only when its recorded source identity/range/speed/target timing still matches current Director intent.

If speed/trim/source changes:

- mark the timing derivative stale;
- do not silently apply the old baked media with new timing;
- reuse the spatial-enhanced base and regenerate only VFI/timing where possible;
- do not rerun ESRGAN/FlashVSR just because speed/FPS changed;
- keep previous successful file until replacement succeeds.

### 21.6 TimelineRender / PreviewPlayer / Hetzner contract

`VideoGenerationTimeline.tsx` must hand TimelineRender an **effective** clip DTO.

For every Director video clip:

```text
url             = active selected variant URL
sceneDuration   = Director timeline duration
startTimeOffset = active variant's effective trim offset
speed           = effective playback speed
```

Effective speed:

```text
timing-baked matching upscale -> 1.0
source/generated              -> segment.speed
non-baked upscale             -> segment.speed
```

`PreviewPlayer` already consumes clip `speed` as media playbackRate, and Hetzner already transcodes speed changes using `setpts=(PTS-STARTPTS)/speed` followed by final render FPS normalization. Therefore no special slow-motion logic belongs in Hetzner if the Director handoff sends the correct effective speed.

Do not mutate persisted `segment.speed` just to satisfy rendering.

---

## 22. Director code changes required in SceneBuilder2

### `src/components/VideoGenerationTimeline.tsx`

Update:

- `activeVisualView` type to include `upscaled`.
- load normalization to recognize `upscaled` and migrate legacy generated+upscaled rows.
- split base generated vs upscale helpers.
- replace `getOriginalVideoForUnit()` with `getPreUpscaleVideoForUnit()` that never returns `upscaledVideoUrl`.
- job completion sets `activeVisualView='upscaled'`, records `upscaledFromView` and timing/source fields.
- re-upscale resolves original generated/uploaded base.
- add Source / Generated / Upscaled selection.
- add Undo Upscale / Use Upscaled / Re-upscale UI.
- preserve new fields in `RemovedMediaSnapshot` and restore flows.
- show timing-baked/stale badge.
- when Trim & Adjust changes source/trim/speed, invalidate timing-baked derivative match rather than double-applying it.

### `vite.director-master-render.ts`

Current logic prefers `upscaledVideoUrl` automatically for generated media. Replace with the same explicit three-variant resolver as Director UI. Preserve existing H3 preview->master protection.

### `src/components/TimelineRender.tsx`

No product-speed mutation. It receives the already resolved active URL/trim/effective speed from Director. Optional metadata may carry original Director speed for debug/display, but render math uses effective speed only.

### `src/components/PreviewPlayer.tsx`

Continue using clip playbackRate. Correctness comes from Director handing it speed 1 for matching baked upscale and original `segment.speed` for source/generated/non-baked upscale.

### `hetzner-render-agent`

No new enhancer-specific speed algorithm is required. Existing planner correctly transcodes whenever speed !=1 or FPS/resolution/codec mismatch. A matching timing-baked derivative may be stream-copy eligible when all final settings match.

### `worker.mjs` / D1

No `project_video_timeline` table migration is needed. Save/hydrate the additive segment JSON fields through existing revisioned `data_json` flow. Enhancer completion must patch the relevant segment atomically/idempotently.

---

## 23. Director writeback completion/idempotency

Completion sequence:

```text
pod uploads object
-> Worker verifies expected R2 object/key/hash/metadata
-> pending_upscales records pending completion metadata
-> Worker patches project_video_timeline segment
-> active selection/provenance fields committed
-> mark pending_upscales completed
```

Never mark completed before authoritative product writeback.

On first successful upscale:

```text
upscaledVideoUrl/ObjectKey = new result
upscaledFromView           = source or generated
upscaledSource...          = exact base identity/range/speed
upscaledTimingBaked        = true/false
upscaledTargetFps          = selected target when VFI used
upscaledOutputDurationMs   = actual output duration
activeVisualView           = upscaled
```

Callbacks are idempotent. Replayed callbacks do not duplicate product refs or charges.

Re-upscale keeps previous upscaled pointer active until replacement writeback succeeds.

Still image active in Director -> `skipped_image`, no pod, no charge, no output mutation.

---

## 24. Video Generation Timeline enhancer UI

Reuse existing Director surfaces; no separate normal-user enhancer page.

```text
Video/Image track header
├── Upscale All
├── Upscale Selected
└── enhancement progress/status

Selected unit Controls
└── Upscale / Enhance tab
```

Upscaler selector:

```text
Topaz Gaia 2
Topaz Proteus
Real-ESRGAN Anime
Real-ESRGAN Real
FlashVSR
```

Resolution:

```text
1080p
1440p / 2K
2160p / 4K
```

Local VFI:

```text
None
RIFE 4.9
GIMM-VFI-F
```

Output FPS:

```text
30
48
60
```

Do not use old `interpolateTo60Fps: boolean` as the product contract.

For speed<1 with RIFE/GIMM, Smooth slow motion is auto-checked. For speed>1 it is off.

Variant controls when available:

```text
Source | Generated | Upscaled
```

Upscaled active state shows Undo Upscale and Re-upscale. Stored but inactive upscale shows Use Upscaled.

---

## 25. Enhancer admin UI

Only render enhancer admin tools for existing `ALLOWED_EMAILS`; API independently enforces authorization.

Placement:

```text
Video Generation Timeline
-> Upscale / Enhance controls
-> Enhancer Admin
```

Runtime/provider controls:

```text
Idle timeout seconds
Priority
Provider Auto | RunPod | Novita
GPU Auto | sm86 | sm89 | sm120 | Exact GPU
Execution Sequential | Parallel
x1/x2/x3/x4/Custom
OOM auto-backoff
active/idle pods
current job
GPU/VRAM
runtime image/digest/build revision
CUDA/PyTorch/CuPy/TensorRT
model precision
loaded model/engine/SHA
telemetry/heartbeat/error/debug
Stop/Delete pod
Manual dispatch
```

TensorRT builder model dropdown exactly:

```text
realesr-animevideov3
realesr-general-x4v3
RIFE 4.9
```

Never image ESRGAN, FlashVSR, GIMM or FILM.

Builder controls:

```text
trusted model/checkpoint/ONNX identity
Precision: FP16
compatibility AMPERE_PLUS | sm86 | sm89 | sm120 | Exact GPU
provider Auto | RunPod | Novita
ESRGAN static shape or RIFE 1080/1440/2160 class
Generate
Validate
Benchmark
Force rebuild
Deactivate
Delete cached inactive engine
Copy R2 key
View logs
```

Inventory exposes model/profile/precision/compatibility/build GPU/provider/TRT/CUDA/file size/build duration/benchmark/timestamps/status.

Encoder admin controls expose NVENC CQ and x265 CRF.

Benchmark UI may compare TRT vs native CUDA, providers, GPU classes, compatibility modes, NVENC vs x265 and native FlashVSR/GIMM throughput.

---

## 26. Provider scheduling and pod lifecycle

RunPod creation mirrors proven H3 operational shape: image, GPU priority/list, `gpuCount=1`, GPU compute, CUDA 13.0-compatible host selection, disk, env/ports, noninterruptible/nonlocked policy.

Novita uses current equivalent product/GPU/rootfs/image/env/ports/network/billing/CUDA filtering fields where exposed.

Enhancer pod env contains only required enhancer worker token/control-plane URL/service kind/R2 access/idle timeout/debug/GPU-only flags/port.

Provider API keys remain control-plane only. Pods receive only scoped runtime credentials.

Forced provider selection never silently crosses provider. Auto may fallback according to policy.

Lifecycle:

```text
priority dispatch
compatible idle reuse
persistent admin idle timeout
provision timeout
job timeout
heartbeat/progress
structured errors/debug
retry/provider fallback
cancel
stale allocation recovery
idle reaper
delete lock
delete verification/retry/backoff
orphan cleanup
```

---

## 27. Pod API and security

Runtime API:

```text
GET  /health
GET  /ready
GET  /capabilities
GET  /telemetry
POST /jobs
GET  /jobs/:id
POST /jobs/:id/cancel
```

Stages:

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
completed
failed
cancelled
```

Security:

- enhancer callback endpoint separate from H3, e.g. `/api/projects/v2/enhancer/pod/events`;
- enhancer HMAC domain `enhancer:${workerId}`;
- master secret may source from `ENHANCER_POD_AUTH_MASTER_SECRET || H3_POD_AUTH_MASTER_SECRET`, but enhancer derivation/state remains separate;
- timestamp + nonce replay protection;
- timing-safe validation;
- callback bound to expected worker + job;
- signed/scoped R2 access;
- server-side allowlists for models/profiles/provider/GPU/precision/encoder settings;
- browser cannot send arbitrary ONNX URLs, engine blobs or shell commands;
- admin APIs enforce `ALLOWED_EMAILS` server-side.

---

## 28. Local video pricing

Use existing SceneBuilder payer/team/credit/refund/idempotency mechanics.

Locked local video pricing:

```text
FAST + no VFI or RIFE 4.9 = 3 credits / billable second
FAST + GIMM-VFI-F         = 5 credits / billable second
QUALITY / FlashVSR        = 10 credits / billable second
```

QUALITY stays 10 credits/sec whether local VFI is None, RIFE or GIMM unless deliberately changed later.

Rules:

- `skipped_image` = 0;
- retries/callback replay never double-charge;
- failure/cancel follows existing refund/reconciliation policy;
- Topaz keeps external premium billing;
- billable duration is the authoritative range/duration actually enhanced;
- timing-baked slow motion bills resulting output duration;
- if selected local VFI is bypassed because no interpolation executes, do not charge the GIMM premium merely because stale UI state selected it.

Example:

```text
5 sec source @ 0.5x -> 10 sec baked output
FAST + RIFE    -> 30 credits
FAST + GIMM-F  -> 50 credits
QUALITY        -> 100 credits
```

---

## 29. Observability and errors

Pod telemetry includes provider/instance/GPU/CC/driver/CUDA/PyTorch/CuPy/TRT/VRAM/GPU utilization/memory/temp/power/NVENC/NVDEC/CPU/RAM/current model/backend/precision/engine/concurrency/dimensions/FPS/stage/progress/model FPS/end-to-end FPS/queue depth/elapsed/ETA when trustworthy.

Per-stage timing:

```text
R2 download
probe/decode
normalization
spatial inference
VFI inference
final conform
encode
audio/remux
R2 upload
TRT cache hit/miss
```

Structured errors include at least:

```text
CUDA_UNAVAILABLE
CUDA_DRIVER_TOO_OLD
CUDA_OOM
GPU_CAPABILITY_MISMATCH
GPU_VRAM_BELOW_POLICY
PRECISION_PARITY_FAILED
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

Include bounded recent log tail and relevant runtime/model/engine/retry metadata.

---

## 30. Build architecture

Enhancer stays isolated under:

```text
enhancer/
  .dockerignore
  docker/
  src/
  models/
  scripts/
```

Do not move enhancer source into root H3 `src/` or enhancer Dockerfiles into root H3 `docker/`.

Conceptual layered DAG:

```text
enhancer-smoke
-> enhancer-base (CUDA 13.0.2)
-> enhancer-torch (PyTorch 2.13 cu130, CuPy cuda13x, TRT 10.14.1.48)
   -> enhancer-vfi-models (RIFE + GIMM)
   -> enhancer-esrgan-models (4 ESRGAN; ONNX only video TRT targets)
      -> enhancer-fast
   -> enhancer-flashvsr-runtime/models
      -> enhancer-quality
```

Model/checkpoint layers precede app code. Remove build/package caches. Generated TRT engines are never Docker layers.

H3 root build workflow and `scripts/remote_build.sh` behavior stay untouched. Enhancer has its own Hetzner workflow and `enhancer/scripts/remote_build.sh` with enhancer-only context.

Docker build != TensorRT build:

```text
Docker -> CPU/Hetzner builder -> runtime/model/ONNX image
TRT    -> compatible NVIDIA GPU pod -> engine -> private R2 + D1
```

Use BuildKit secrets for private model/HF downloads. Runtime credentials are injected only at pod creation. Publish immutable git-SHA image tag/digest; aliases may exist but jobs record exact runtime digest.

Docker success is not GPU qualification. sm86/sm89/sm120/AMPERE_PLUS smoke tests are separate release gates.

---

## 31. Repository rename

Approved target repository identity:

```text
khuzaimamussawar/GPU-runtime
```

Rename is repository identity only. Do not rename H3 Docker Hub images, H3 D1 tables, task families, APIs or lifecycle identifiers.

Audit established:

- maintained H3 Hetzner workflow passes dynamic `${GITHUB_REPOSITORY}`;
- H3 remote build clones `${GITHUB_REPOSITORY}` dynamically;
- `/opt/minimax-h3-serverless` is only a local checkout path and may remain initially;
- `GH_FH_TOKEN_MM_H3_SERVERLESS` may retain its secret name initially;
- enhancer workflow also uses dynamic repository identity;
- SceneBuilder2 `minimax-h3-*` references are primarily Docker/runtime service identities, not GitHub clone dependencies;
- no audited external `uses: khuzaimamussawar/minimax-h3-serverless/...@...` dependency was found.

After rename update documentation/developer remotes. Do not rebuild H3 images solely because the GitHub repository name changed.

---

## 32. Source/license policy

```text
Real-ESRGAN -> pinned permissive official upstream + deterministic BasicSR compatibility patch
RIFE        -> permissive upstream; our own TRT adapters/exporters
FlashVSR    -> pin/review official/permissive attention/runtime dependencies
GIMM-VFI-F  -> production use remains commercial-license/permission gated
Topaz       -> external licensed backend; no local redistribution
```

Noncommercial community TensorRT wrappers may be references/benchmarks only; do not copy incompatible code into the commercial runtime.

Primary precision references:

```text
Real-ESRGAN official inference: FP16 default, --fp32 override
RIFE official/Practical-RIFE: explicit FP16 inference support
FlashVSR v1.1 official full inference: torch.bfloat16 model/input path
GIMM-VFI-F official: no canonical BF16/FP16 production contract -> FP32 V1
```

---

## 33. Qualification / release gates

Before production eligibility on each GPU/runtime class verify:

```text
CUDA 13.0.2
PyTorch 2.13.0 cu130
CuPy kernel
TensorRT 10.14.1.48 builder/runtime where applicable
GPU-only invariant
precision matrix + parity
native Storyboard ESRGAN FP16
native video ESRGAN FP16
video ESRGAN TRT build/load/parity
RIFE native + TRT with non-0.5 timestep values
RIFE 2160 FP16 artifact check
GIMM-F FP32 arbitrary timestep CUDA/CuPy
FlashVSR BF16 quality/runtime
scene-cut behavior
no static-frame skip behavior
30/48/60 scheduling from varied CFR/VFR sources
timing-baked 0.5x/0.6x/0.75x slow motion
Undo Upscale + Use Upscaled + Re-upscale source correctness
source/generated/upscaled active selection
TimelineRender effective-speed handoff
NVDEC/NVENC
R2/auth/callback/idempotency
credit charge/refund behavior
```

Golden inputs include realistic/anime faces, hair/detail, text, foliage, gradients, camera pans, fast motion, occlusion, hard cuts, portrait/landscape, H3-origin rasters and slightly off-ratio inputs.

Measure inference/end-to-end FPS, peak VRAM, CPU/RAM, decode/encode, R2 transfer and visual/parity quality.

---

## 34. Final canonical capability matrix

```text
FAST
  CUDA 13.0 Update 2
  PyTorch 2.13.0 cu130
  CuPy cupy-cuda13x
  TensorRT 10.14.1.48 builder + runtime
  FFmpeg NVDEC/NVENC

  IMAGE
    RealESRGAN_x4plus_anime_6B -> FP16 native GPU PyTorch only
    RealESRGAN_x4plus          -> FP16 native GPU PyTorch only

  VIDEO
    realesr-animevideov3       -> FP16 TRT preferred; FP16 native GPU fallback
    realesr-general-x4v3       -> FP16 TRT preferred; FP16 native GPU fallback

  VFI
    RIFE 4.9                   -> FP16 TRT preferred; FP16 native fallback; GPU FP32 safety fallback only if exact profile fails parity
    GIMM-VFI-F                 -> FP32 PyTorch + CuPy; >=24 GB VRAM; no TRT

QUALITY
  CUDA 13.0 Update 2
  PyTorch 2.13.0 cu130
  CuPy cupy-cuda13x
  TensorRT 10.14.1.48 runtime for RIFE
  FFmpeg NVDEC/NVENC

  VIDEO
    FlashVSR v1.1              -> BF16 native GPU; no TensorRT

  VFI
    RIFE 4.9                   -> compatible FP16 TRT preferred
    GIMM-VFI-F                 -> FP32 PyTorch + CuPy

OUTPUT FPS
  30 / 48 / 60 only
  arbitrary source FPS/VFR accepted

TRT ENGINE SCOPE
  realesr-animevideov3
  realesr-general-x4v3
  RIFE 4.9

TRT BASELINE CACHE
  15 sm86 + 15 sm89 + 15 sm120 + 15 AMPERE_PLUS = 60 engines

DIRECTOR MEDIA VARIANTS
  source
  generated pre-upscale master
  upscaled

SLOW-MO
  speed<1 + RIFE/GIMM -> Smooth slow motion auto-check
  baked upscale active -> effective render speed 1
  source/generated active -> original Director speed returns
  non-baked upscale active -> original Director speed remains
```

**No FILM. No Storyboard image ESRGAN TensorRT. No FlashVSR TensorRT. No GIMM TensorRT. No FP8 V1. No neural CPU fallback. No recursive re-upscale from `upscaledVideoUrl`.**

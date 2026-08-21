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
enhancer_engine_builds
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

Admin TensorRT engine creator jobs use this FAST image with service kind `enhancer_engine_builder`. They do not run product image/video upscales, do not write Director media, and never fabricate `.engine` files. Missing trusted ONNX/checkpoint/TensorRT tooling fails the engine-build job rather than producing a dummy artifact.

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

The enhancer torch layer installs pinned `torch` + `torchvision` only. Do not add `torchaudio` unless runtime code starts importing it and a matching cu130 wheel exists; enhancer audio preservation/remux uses ffmpeg/AV tooling, not TorchAudio. The TensorRT Python package install uses NVIDIA's `10.14.1.48.post1` pip publication because that is the matching Linux cu13 binding publish for the TensorRT `10.14.1.48` runtime line.

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

Exclude all MIG/partitioned GPU products. Also exclude A100/A800/H100/H200/B200 and other high-cost product-outlier classes even if technically usable.

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
-> enhancer_engine_builds action=generate
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

D1 stores metadata/control state, not engine binary bytes. `pending_upscales` remains the unified V1 product-work job/history table. `enhancer_engine_builds` is the admin engine inventory/source of truth; no parallel product-media table is introduced for engines.

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

One unified product-work operational job table for:

```text
image_upscale
video_upscale
vfi_only
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

Storyboard image upscale rows persist exact source URL/object key, source width/height, target, model/backend, selection scope, batch id/index/total and output object key. `POST /api/upscale-batch` / Storyboard Upscale All must create these local enhancer rows and must not insert legacy Replicate-shaped `pending_upscales` rows.

Engine builder/admin rows are separate operational admin inventory rows in `enhancer_engine_builds`. They store exact R2 key/SHA/size, ONNX/checkpoint hash, TRT/CUDA, FP16, compatibility target, CC/exact GPU, profile, builder identity, validation/benchmark and active state. They are not product media truth and do not write Storyboard or Director media.

Keep separate:

```text
enhancer_engine_builds
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
smoke
-> base (CUDA 13.0.2)
   -> torch (PyTorch 2.13 cu130, CuPy cuda13x, TRT 10.14.1.48)
      -> vfi-models (RIFE + GIMM)
         -> FAST branch
            -> esrgan-models (image ESRGAN checkpoints + video ESRGAN checkpoint/export source)
               -> fast        # final Docker image: scenebuilder-enhancer-fast
         -> QUALITY branch
            -> flashvsr-runtime
               -> flashvsr-models
                  -> quality  # final Docker image: scenebuilder-enhancer-quality
```

FAST and QUALITY are siblings after the shared base/VFI layers; `fast` is not a parent of `quality`. Model/checkpoint/export-source layers precede app code. Remove build/package caches. Generated TRT engines are never Docker layers.

H3 root build workflow and `scripts/remote_build.sh` behavior stay untouched. Enhancer has its own Hetzner workflow and `enhancer/scripts/remote_build.sh` with enhancer-only context.

The enhancer workflow defaults to a Hetzner `cpx32` CPU builder. Branch aliases expand before build:

```text
target=fast    -> smoke,base,torch,vfi-models,esrgan-models,fast
target=quality -> smoke,base,torch,vfi-models,flashvsr-runtime,flashvsr-models,quality
```

`target=all` expands to the shared layers, then the FAST branch, then the QUALITY branch:

```text
smoke,base,torch,vfi-models,esrgan-models,fast,flashvsr-runtime,flashvsr-models,quality
```

`target=remaining` is the continuation path after a partial/failed run: pass the same `image_tag` and optional previous workflow run id; the remote script checks Docker Hub for already-pushed layer images and builds only the missing targets.

Docker build != TensorRT build:

```text
Docker -> CPU/Hetzner builder -> shared base + FAST/QUALITY runtime/model/export-source images
TRT    -> compatible NVIDIA GPU FAST engine-builder pod -> .engine -> private R2 + D1
```

FAST is the only engine-builder image. It generates the 15-engine set per selected compatibility target for the exact approved TRT scope: `realesr-animevideov3`, `realesr-general-x4v3`, and `RIFE 4.9`. QUALITY may consume compatible active RIFE engines at runtime, but it does not build engines. Neither image creates TensorRT for Storyboard image ESRGAN, FlashVSR, GIMM or FILM.

Use BuildKit secrets for private model/HF downloads. Runtime credentials are injected only at pod creation. Publish immutable git-SHA image tag/digest; aliases may exist but jobs record exact runtime digest.

Docker success is not GPU qualification. sm86/sm89/sm120/AMPERE_PLUS smoke tests are separate release gates.

Current implementation checkpoint:

- `engine_builder.py` requires `trtexec`; the Docker layer must install/prove that CLI, not only Python `tensorrt` import.
- The Docker model layers bake SHA-256-verified ONNX artifacts for `realesr-animevideov3`, `realesr-general-x4v3`, and `rife-4.9` under `/opt/scenebuilder-models/onnx/`.
- Baked ONNX artifacts are not proof of activation. The engine-builder/admin flow must still record provenance and pass PyTorch/native-vs-ONNX-vs-TRT parity before activating generated engines.
- If an ONNX path is missing or hash verification fails, the image build or engine-build job must fail; it must not generate a dummy engine.

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

**No FILM. No Storyboard image ESRGAN TensorRT. No FlashVSR TensorRT. No GIMM TensorRT. No FP8 V1. No neural CPU fallback. No recursive re-upscale from `upscaledVideoUrl`. No MIG/partitioned GPUs.**

---

## 35. Shared H3 variables/secrets are a hard compatibility contract

This section **supersedes any earlier enhancer wording that proposes `ENHANCER_*` copies of credentials/configuration already used by H3 for the same purpose.** The enhancer must reuse the existing H3/SceneBuilder names exactly wherever the semantic purpose is the same. Do not create a second set of secrets merely because the workload is called enhancer.

### 35.1 Audited existing Worker/control-plane names

SceneBuilder's current H3 pod control plane already resolves and/or uses these names:

```text
R2_PUBLIC_URL
R2_ENDPOINT
R2_BUCKET_NAME
R2_REGION

H3_POD_R2_ACCESS_KEY
H3_POD_R2_SECRET_KEY
H3_POD_AUTH_MASTER_SECRET

RUNPOD_API_KEY
NOVITA_API_KEY

WORKER_ORIGIN
ALLOWED_EMAILS
```

The existing H3 R2 resolver also recognizes these already-established H3-prefixed overrides when present:

```text
H3_POD_R2_BUCKET_NAME
H3_POD_R2_ENDPOINT
H3_POD_R2_REGION
```

and resolves credentials as the current H3 implementation does, including the existing shared `R2_*` fallback names where applicable. Enhancer control-plane code must call/reproduce this same resolution contract rather than inventing enhancer-specific R2 variables.

There is **no new `R2_ENDPOINT_ID` variable**. The existing endpoint configuration name is exactly:

```text
R2_ENDPOINT
```

### 35.2 Pod runtime env names are reused exactly

The H3 control plane currently injects these generic/shared runtime names into a provisioned pod:

```text
R2_BUCKET_NAME
R2_ENDPOINT
R2_ACCESS_KEY
R2_SECRET_KEY
R2_REGION
R2_PUBLIC_URL

SCENEBUILDER_POD_TOKEN
SCENEBUILDER_WORKER_ID
SCENEBUILDER_CONTROL_URL

H3_POD_IDLE_TIMEOUT_SECONDS
H3_POD_PORT
```

Enhancer FAST/QUALITY pods must read the **same names** for the same concepts. Do not add:

```text
ENHANCER_R2_BUCKET_NAME
ENHANCER_R2_ENDPOINT
ENHANCER_R2_ACCESS_KEY
ENHANCER_R2_SECRET_KEY
ENHANCER_R2_PUBLIC_URL
ENHANCER_POD_TOKEN
ENHANCER_WORKER_ID
ENHANCER_CONTROL_URL
ENHANCER_POD_IDLE_TIMEOUT_SECONDS
ENHANCER_POD_PORT
```

The enhancer Docker image is already a different runtime image, so it knows that `SCENEBUILDER_CONTROL_URL` should be combined with the enhancer callback/API path. No extra service-kind secret is required just to rename the same control-plane origin/token concepts.

### 35.3 One auth master secret, domain-separated derived tokens

Use exactly the existing Worker secret:

```text
H3_POD_AUTH_MASTER_SECRET
```

for both H3 and enhancer pod-token derivation. **Do not create or read `ENHANCER_POD_AUTH_MASTER_SECRET`.**

Cross-service token confusion is prevented by derivation-domain separation, not by duplicating master secrets:

```text
H3 derived token      -> existing H3 derivation contract
enhancer derived token -> HMAC/HKDF message/domain includes `enhancer:${workerId}`
```

The derived token delivered to either pod remains named:

```text
SCENEBUILDER_POD_TOKEN
```

Enhancer callback nonce tables, worker/job binding, replay state and endpoint remain enhancer-specific even though the master secret is shared.

### 35.4 R2 credentials and storage identity are shared, not duplicated

Enhancer uses the same SceneBuilder R2 account/bucket identity already used by H3 and product media. Current audited names remain authoritative:

```text
Worker/control plane:
  R2_BUCKET_NAME
  R2_ENDPOINT
  R2_REGION
  R2_PUBLIC_URL
  H3_POD_R2_ACCESS_KEY
  H3_POD_R2_SECRET_KEY

Pod runtime:
  R2_BUCKET_NAME
  R2_ENDPOINT
  R2_REGION
  R2_PUBLIC_URL
  R2_ACCESS_KEY
  R2_SECRET_KEY
```

No enhancer-only R2 access-key pair, bucket-name variable, endpoint variable or public-URL variable is introduced.

Authorization is still least-privilege by request/job/object-key validation. Sharing the credential **name/value source** does not mean allowing arbitrary object access in application logic.

### 35.5 Provider API keys are shared control-plane secrets

Use the existing names exactly:

```text
RUNPOD_API_KEY
NOVITA_API_KEY
```

H3 and enhancer provision different pod images/state tables but use the same provider accounts/credentials. Do not create:

```text
ENHANCER_RUNPOD_API_KEY
ENHANCER_NOVITA_API_KEY
```

Provider API keys remain in the SceneBuilder control plane and are **never injected into GPU pods**.

### 35.6 Build/repository/model-download secrets are reused exactly

The maintained H3 Hetzner Docker workflow already uses:

```text
HETZNER_TOKEN
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
HF_TOKEN
GH_FH_TOKEN_MM_H3_SERVERLESS
```

These exact GitHub secret names are the enhancer build contract too. Do not add enhancer-prefixed equivalents.

The enhancer workflow reuses:

```text
HETZNER_TOKEN
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
GH_FH_TOKEN_MM_H3_SERVERLESS
```

`.github/workflows/hetzner-enhancer-build.yml` and `enhancer/scripts/remote_build.sh` pass/use the existing:

```text
HF_TOKEN
```

with the **same name** used by the H3 build. Private Hugging Face/model downloads use BuildKit secret mounts or equivalent non-layer secret injection; never bake the token into image layers or logs.

`GH_FH_TOKEN_MM_H3_SERVERLESS` intentionally keeps its existing name even after the approved future repository rename to `GPU-runtime`; do not duplicate/rename the secret solely because the repository name changes.

### 35.7 Existing Cloudflare/bootstrap credential is not duplicated

The audited H3 Worker-secret bootstrap workflow uses the existing GitHub secret:

```text
SCENEBUILDER2_D1_R2_ACCESS
```

If enhancer-related bootstrap/deploy automation needs the same Cloudflare permission, reuse this exact existing secret rather than creating an enhancer-specific Cloudflare token. It is not a GPU pod runtime variable.

### 35.8 `ALLOWED_EMAILS` stays shared

Enhancer admin/API authorization uses the existing:

```text
ALLOWED_EMAILS
```

Do not introduce `ENHANCER_ALLOWED_EMAILS`. H3 admin controls and enhancer admin controls may expose different operations, but authorization comes from the same server-side allowlist unless the product deliberately changes that policy later.

### 35.9 Build-only vs control-plane-only vs pod-runtime exposure

Reusing a secret name does **not** mean spraying every secret into every environment. Preserve least exposure:

```text
GitHub/Hetzner build only:
  HETZNER_TOKEN
  DOCKERHUB_USERNAME
  DOCKERHUB_TOKEN
  HF_TOKEN
  GH_FH_TOKEN_MM_H3_SERVERLESS

SceneBuilder control plane only:
  RUNPOD_API_KEY
  NOVITA_API_KEY
  H3_POD_AUTH_MASTER_SECRET
  H3_POD_R2_ACCESS_KEY
  H3_POD_R2_SECRET_KEY
  SCENEBUILDER2_D1_R2_ACCESS   # only bootstrap/deploy tooling that needs it

Pod runtime, injected from the existing H3 resolution path:
  R2_BUCKET_NAME
  R2_ENDPOINT
  R2_ACCESS_KEY
  R2_SECRET_KEY
  R2_REGION
  R2_PUBLIC_URL
  SCENEBUILDER_POD_TOKEN
  SCENEBUILDER_WORKER_ID
  SCENEBUILDER_CONTROL_URL
  H3_POD_IDLE_TIMEOUT_SECONDS
  H3_POD_PORT
```

Do not inject DockerHub/GitHub/Hugging Face/provider API/master-auth secrets into customer inference pods merely to satisfy the name-reuse rule.

### 35.10 Names that are intentionally *not* shared because the value/meaning differs

The no-duplication rule applies when the semantic purpose and value are the same. It does **not** force enhancer to point at H3-specific runtime artifacts/settings.

For example:

```text
H3_POD_IMAGE
H3_TARGET_OUTPUT_SECONDS_PER_WORKER
```

remain H3-specific because enhancer uses different Docker images and workload sizing. Enhancer image identity should come from its own fixed/runtime configuration already required by the enhancer architecture, **not** by overloading `H3_POD_IMAGE` with two simultaneous values.

This is the only acceptable reason to introduce enhancer-specific configuration: the value or semantic meaning is genuinely different. Never duplicate a credential/config merely for naming symmetry.

### 35.11 Implementation/review gate

Before merge, CI/static review must fail if new enhancer code introduces same-purpose duplicate names such as:

```text
ENHANCER_POD_AUTH_MASTER_SECRET
ENHANCER_R2_*
ENHANCER_RUNPOD_API_KEY
ENHANCER_NOVITA_API_KEY
ENHANCER_HF_TOKEN
ENHANCER_DOCKERHUB_*
ENHANCER_GITHUB_TOKEN
ENHANCER_ALLOWED_EMAILS
```

Required integration tests must prove:

```text
H3 jobs still resolve their existing env/secrets unchanged
enhancer jobs resolve the same shared sources
enhancer pod receives the same generic runtime R2/token env names
H3 and enhancer derived pod tokens are domain-separated
provider keys never reach pods
build-only credentials never reach runtime pods
R2 upload/download works for both services
no secret value is logged
```

**Hard rule:** same goal + same credential/configuration source = same existing H3/SceneBuilder name. No duplicate enhancer secret namespace.

---

## 36. Exact provider GPU allowlists, family mapping and routing priority

This section supersedes the representative examples in Section 5 wherever a concrete provider allowlist or priority is needed. Provider inventory is still queried live, but the scheduler may only choose a SKU on these allowlists and only after boot-time CUDA/VRAM/runtime qualification succeeds.

### 36.1 Canonical family mapping

The V1 compute-family identities used by runtime routing and TensorRT engine administration are:

```text
Ampere    -> sm_86
Ada       -> sm_89
Blackwell -> sm_120
```

For the RunPod SKU labels seen in the current provider inventory, use the following canonical mapping.

**Ampere / sm86:**

```text
RTX A4000      16 GB
RTX A4500      20 GB
RTX A5000      24 GB
RTX 3090       24 GB
A40            48 GB
RTX A6000      48 GB
```

**Ada / sm89:**

```text
RTX 2000 Ada   16 GB
RTX 4000 Ada   20 GB
L4             24 GB
RTX 4090       24 GB
L40            48 GB
RTX 6000 Ada   48 GB
L40S           48 GB
```

**Blackwell / sm120:**

```text
RTX PRO 4000 / RTX PRO 4000 Blackwell               24 GB
RTX 5090                                             32 GB
RTX PRO 4500 / RTX PRO 4500 Blackwell               32 GB
RTX PRO 4500 SE / RTX PRO 4500 Blackwell Server Ed. 32 GB
```

The provider-facing matcher must tolerate RunPod display-name variations while normalizing to one canonical internal SKU/family.

**MIG/partitioned GPUs are always excluded**, even when their reported VRAM is <=48 GB or their underlying GPU family is otherwise supported. Provider inventory entries containing `MIG`, or otherwise reported as partitioned/sliced instances, are rejected before scheduling.

Do not include GPUs above the global 48 GB cap. In particular, screenshots may expose A100/H100 and other 80+ GB products; they remain ineligible for enhancer V1 even if technically compatible.

### 36.2 RunPod FAST / Real-ESRGAN automatic priority

For FAST Real-ESRGAN jobs, the requested traversal order is intentionally **not simple ascending VRAM**. The locked VRAM-tier order is:

```text
16 GB
20 GB
24 GB
48 GB
32 GB
```

Within every VRAM tier, family priority is:

```text
Ada
Ampere
Blackwell
```

Therefore the routing matrix is:

| Tier | Ada first | Ampere second | Blackwell third |
|---|---|---|---|
| 16 GB | RTX 2000 Ada | RTX A4000 | — |
| 20 GB | RTX 4000 Ada | RTX A4500 | — |
| 24 GB | L4, RTX 4090 | RTX A5000, RTX 3090 | RTX PRO 4000 |
| 48 GB | L40, RTX 6000 Ada, L40S | A40, RTX A6000 | — |
| 32 GB | — | — | RTX 5090, RTX PRO 4500, RTX PRO 4500 SE |

`48 GB -> 32 GB` is deliberate and must not be “corrected” to numeric ascending order.

Within one exact tier + family cell, the scheduler may use live provider availability/capacity and price to choose among the listed SKUs. It must not silently jump to a later tier/family while an earlier eligible candidate is available and qualified.

Before applying this traversal, filter by the job's real minimum requirements. Examples:

```text
FAST ESRGAN ordinary job -> may start at 16 GB when the exact model/resolution/concurrency tuple is qualified
FAST + GIMM-F            -> remove 16/20 GB; physical VRAM >=24 GB is mandatory
QUALITY / FlashVSR       -> apply the separate 4090-class-or-better quality floor and benchmarked VRAM rules
known unsafe/OOM tuple   -> start at the next known-safe VRAM tier instead of deliberately repeating a known failure
```

OOM escalation continues to use the existing xN backoff first, then moves to a larger eligible GPU according to the allowed routing policy.

### 36.3 Novita enhancer allowlist and exact priority

For enhancer V1, Novita is intentionally restricted to these four SKUs only, in this exact priority order:

```text
1. RTX 4090       24 GB  Ada / sm89
2. RTX 6000 Ada   48 GB  Ada / sm89
3. L40S           48 GB  Ada / sm89
4. RTX 5090       32 GB  Blackwell / sm120
```

Do not add other Novita GPU products merely because the provider exposes them. This list is separate from RunPod's broader allowlist.

A forced Novita request never crosses to RunPod. `Auto` may fall back to RunPod after Novita candidates are exhausted, subject to the requested compute-family constraint.

### 36.4 Admin TensorRT engine-family selection provisions the requested family

Enhancer Admin TensorRT controls must show clear family labels:

```text
AMPERE_PLUS portable
Ampere (sm86)
Ada (sm89)
Blackwell (sm120)
Exact GPU
```

When the admin clicks a same-compute-family target, that selection is a **hard architecture constraint**, not a suggestion.

RunPod family candidates are:

```text
Ampere (sm86)
  RTX A4000
  RTX A4500
  RTX A5000
  RTX 3090
  A40
  RTX A6000

Ada (sm89)
  RTX 2000 Ada
  RTX 4000 Ada
  L4
  RTX 4090
  L40
  RTX 6000 Ada
  L40S

Blackwell (sm120)
  RTX PRO 4000
  RTX 5090
  RTX PRO 4500
  RTX PRO 4500 SE
```

The control plane first applies the engine-build job's minimum safe VRAM/profile requirement, then chooses an available SKU **inside that family only**. For example, if a future 2160 RIFE builder profile is qualified only at >=24 GB, the scheduler must skip 16/20 GB members of the selected family rather than trying them first.

Provider behavior for family builds:

```text
Provider = RunPod
  -> choose only from the RunPod list for the selected family

Provider = Novita
  Ada       -> RTX 4090 -> RTX 6000 Ada -> L40S
  Blackwell -> RTX 5090 only
  Ampere    -> no eligible Novita SKU in V1; return provider/family capacity-policy failure

Provider = Auto
  -> may move between providers only while preserving the selected family
```

Never satisfy `Ada (sm89)` with an Ampere or Blackwell builder just because it is available. Same rule for the other families.

For `Exact GPU`, no family substitute is allowed: provision that exact **full physical non-MIG SKU** or leave the build queued/fail according to admin policy. An exact MIG/partitioned SKU request is rejected as out of policy.

### 36.5 AMPERE_PLUS builder host

The portable `AMPERE_PLUS` TensorRT set should be generated on an approved **Ampere sm86 RunPod full GPU** by default, using the intended TensorRT hardware-compatibility flags, then validated on representative Ada and Blackwell full GPUs before activation.

The default builder may move among the approved Ampere RunPod SKUs according to minimum build VRAM and availability. Do not require a separate engine artifact per Ampere SKU for the portable set.

### 36.6 MIG / partitioned GPU hard exclusion

MIG and other provider-partitioned GPU slices are **not part of the enhancer fleet at all**.

This exclusion applies to every enhancer operation:

```text
Storyboard image upscale
Director video upscale
RIFE
GIMM-F
FlashVSR
TensorRT engine build
TensorRT engine validation
TensorRT benchmark
Exact GPU admin dispatch
manual/admin pod dispatch
```

Examples explicitly excluded from current RunPod inventory include:

```text
PRO 6000 MIG 24GB
PRO 6000 MIG 48GB
```

The provider matcher must reject a candidate when any authoritative provider/runtime signal indicates MIG or partitioning. Do not rely only on display-name matching when provider metadata exposes partition/full-GPU state.

No MIG fallback is permitted when a full GPU of the requested family is unavailable. Continue to the next approved full-GPU candidate/provider while preserving forced provider/family semantics; otherwise queue/fail with capacity-policy status.

### 36.7 D1/admin representation

The canonical allowlists are code-reviewed policy, while `enhancer_config.gpu_policy_json` stores approved enable/disable/order overrides and operational tuning. Admin UI may reorder or disable known allowed SKUs but may not authorize an unknown/out-of-policy GPU or any MIG/partitioned GPU directly from the browser.

Persist actual routing decisions in enhancer operational state:

```text
pending_upscales.requested_provider
pending_upscales.actual_provider
pending_upscales.requested_gpu
pending_upscales.actual_gpu
pending_upscales.settings_json       # requested family / admin routing snapshot

enhancer_pod_workers.gpu_class
enhancer_pod_workers.provider_gpu_name
enhancer_pod_workers.compute_capability
enhancer_pod_workers.vram_mb
```

The admin panel must display both the requested family/SKU and the actual provisioned provider GPU so a fallback is never invisible.

### 36.8 Release gate for every allowlisted SKU

Being on this list makes a GPU **schedulable candidate**, not automatically production-qualified. Each SKU still needs the relevant boot/release checks:

```text
full physical GPU / non-MIG verification
CUDA 13.0 compatibility
correct compute capability
physical VRAM check
PyTorch/CuPy smoke
TensorRT load/build where applicable
model smoke/parity
GIMM >=24 GB rule
FlashVSR capability/VRAM rule
NVDEC/NVENC capability where video hardware codec is expected
OOM/concurrency safe tuple data
```

Failed qualification marks that SKU/runtime tuple unhealthy and the control plane tries the next candidate allowed by the same job/provider/family policy.

---

## 37. Director D1/media-selection hard contract

This section is the final tie-breaker for Director media naming, active selection, re-upscale input, Undo Upscale and TimelineRender handoff. It **does not rename existing SceneBuilder2 D1/product fields** and does not introduce a second Director media table.

### 37.1 Keep existing D1/product field names exactly

`project_video_timeline` remains physically unchanged:

```text
id
project_id
batch_index
data_json
revision
created_at
updated_at
```

All Director media state remains inside each segment in revisioned `data_json`. Do **not** rename existing media keys just to make enhancer terminology cleaner.

Keep the established keys and their object-key partners:

```text
imageUrl
sourceVideoUrl
sourceVideoObjectKey
videoUrl
videoObjectKey
renderVideoUrl
renderVideoObjectKey
upscaledVideoUrl
upscaledVideoObjectKey
activeVisualView
```

Do not add a duplicate `activeVideoUrl`, `currentVideoUrl`, `originalVideoUrl`, `previewVideoUrl`, or another top-level D1 media pointer. A duplicated active URL can drift away from the real source/generated/upscaled pointers. `activeVisualView` is the selector; the existing URL/object-key fields are the assets.

`upscaledVideoUrl` / `upscaledVideoObjectKey` are populated only after a valid upscale successfully uploads and authoritative writeback succeeds. If no successful upscale exists, those fields may be absent and `activeVisualView='upscaled'` is invalid.

### 37.2 Existing URL/object-key roles

Use the existing names with these semantics:

```text
imageUrl
  -> source image when the Director segment is image-based

sourceVideoUrl / sourceVideoObjectKey
  -> uploaded/original source video retained independently of generation/upscale

videoUrl / videoObjectKey
  -> existing generated browser-playable/generated asset under the current Director/provider contract
     (for H3 this can be the lightweight H.264 preview)

renderVideoUrl / renderVideoObjectKey
  -> authoritative generated/render master when present
     (for H3 this is the H.265/master side of the existing preview/master contract)

upscaledVideoUrl / upscaledVideoObjectKey
  -> enhancer derivative only; never the definition of "generated"
```

Do not blindly assume field names prove codec/master quality for every legacy row. Current Director master-resolution logic already has to protect against legacy rows where `renderVideoUrl` can point at an H.264 preview while `renderVideoObjectKey` identifies the true master. `baseGeneratedMediaForSegment()` must preserve that existing canonical preview->master protection.

The important invariant is not "always use renderVideoUrl". It is:

```text
generated selection / enhancer generated input
-> resolve the authoritative pre-upscale generated master using existing canonical Director logic
-> NEVER resolve through upscaledVideoUrl
```

### 37.3 Mixed uploaded + generated projects are per-segment, not project-wide

One Director project may contain any mixture of:

```text
uploaded-video segments
generated-video segments
segments that retain both uploaded source + generated result
segments with a successful upscale
image segments
```

Do not introduce a project-wide "uploaded" or "generated" mode. Resolve every visual segment independently.

For a segment that has both uploaded and generated video:

```text
activeVisualView = source
  -> use sourceVideoUrl/sourceVideoObjectKey

activeVisualView = generated
  -> use authoritative generated pre-upscale master

activeVisualView = upscaled
  -> use upscaledVideoUrl only when it exists and is valid
```

Selecting one variant never deletes the other variants.

### 37.4 `activeVisualView` is the only active-asset selector

Allowed values:

```ts
activeVisualView?: 'source' | 'generated' | 'upscaled';
```

Validity rules:

```text
source
  valid when a source image/uploaded source video exists

generated
  valid when a base generated/render video exists

upscaled
  valid only when a successful upscaledVideoUrl exists
```

When loading data, if `activeVisualView='upscaled'` but no valid upscale exists, normalize back to the best existing pre-upscale variant rather than inventing an active URL.

Do not auto-switch to `upscaled` merely because an old/possibly stale upscale URL exists when the user has explicitly selected Source or Generated.

### 37.5 Re-upscale truth table: never upscale an upscale

This is a hard neural-input invariant:

| Current active view | Recorded upscale origin | Re-upscale neural input |
|---|---|---|
| `source` | any | uploaded/source video |
| `generated` | any | authoritative generated pre-upscale master |
| `upscaled` | `source` | uploaded/source video recorded by `upscaledFromView` + source identity |
| `upscaled` | `generated` | authoritative generated pre-upscale master recorded by `upscaledFromView` + source identity |

Forbidden:

```text
upscaledVideoUrl -> ESRGAN / FlashVSR -> another upscale
```

This remains forbidden for:

```text
Re-upscale one clip
Upscale Selected -> Re-upscale all selected
Upscale All -> including already-upscaled
retry after failed replacement
stale-upscale replacement
admin/manual dispatch
```

For legacy upscales without explicit `upscaledFromView`, infer the base once from the retained pre-upscale assets:

```text
valid generated master exists -> generated
else valid uploaded source video exists -> source
else do not queue recursive upscale
```

The enhancer request must persist `source_view`, exact source URL/object key and source timing in `pending_upscales`, so the job itself proves that its input was pre-upscale media.

### 37.6 Undo Upscale is selection, not deletion

After an upscale succeeds, retain all original pointers:

```text
sourceVideoUrl / imageUrl         retained
generated videoUrl/renderVideoUrl retained
upscaledVideoUrl                  added
```

Normal Undo Upscale does exactly:

```text
activeVisualView = upscaledFromView
```

It does **not**:

```text
clear upscaledVideoUrl
delete the R2 upscale
replace sourceVideoUrl
replace renderVideoUrl/videoUrl
refund the upscale
```

Therefore the user can move among the variants whenever they exist:

```text
Source
Generated
Upscaled
```

When Upscaled is inactive but still stored, expose `Use Upscaled`. When Upscaled is active, expose `Undo Upscale` and `Re-upscale`.

### 37.7 Successful upscale writeback and replacement behavior

First successful upscale:

```text
write upscaledVideoUrl/upscaledVideoObjectKey
record upscaledFromView = source | generated
record exact pre-upscale source identity/timing fields
set activeVisualView = upscaled
```

Do not set `activeVisualView='upscaled'` before the new output exists and authoritative D1 writeback succeeds.

Re-upscale is replacement-from-base, not chaining:

```text
old successful upscale remains active
-> queue new job from source/generated pre-upscale master
-> new job uploads/verifies
-> atomically replace upscaledVideoUrl + provenance
-> keep activeVisualView=upscaled
```

If replacement fails or is cancelled, the previous successful upscale remains intact and selectable.

### 37.8 Video Generation Timeline owns selection; TimelineRender consumes the resolved DTO

The product flow remains:

```text
project_video_timeline data_json
        ↓
VideoGenerationTimeline.tsx
  resolves Source / Generated / Upscaled
  resolves authoritative generated master vs preview
  resolves trim + effective speed
        ↓
TimelineRender
```

Do not change this ownership.

`TimelineRender` must not query D1, reinterpret `sourceVideoUrl`/`videoUrl`/`renderVideoUrl`/`upscaledVideoUrl`, or automatically prefer an upscale. `VideoGenerationTimeline.tsx` hands it the already resolved clip DTO:

```text
url
sceneDuration
startTimeOffset
speed
```

This is especially important for mixed projects containing uploaded and generated clips in the same render timeline. Each clip arrives already resolved from its own Director segment state.

### 37.9 Existing Director UI shell is the required enhancer shell

Do not add another normal-user enhancer page, timeline track, toolbar, clip-selection list, settings-card design or modal design.

Reuse exactly:

```text
Controls row
  [Video] [Upscale]

Upscale controls
  existing per-unit card
  Model
  Output
  VFI
  FPS
  Smooth slow motion
  existing per-control Apply all behavior

Video / Image track
  no selection -> Generate All / Upscale All
  selection    -> Deselect All / Generate Selected / Upscale Selected
  same selectedUnitIds for both operations

Per-clip primary action:
  Video controls + Generate enabled intent  -> Generate / Regenerate
  Upscale controls + upscale enabled intent -> Upscale / Re-upscale
  Upscale controls + upscale disabled       -> leave existing Generate / Regenerate behavior

existing bulk dialog shell
  Generate -> Remaining / All
  Upscale  -> Remaining / Re-upscale All
```

Current implementation audit requirement before enhancer UI completion:

- `vite.director-video-remaining-fix.ts` contains the intended tightened Generate Remaining semantics but is not currently registered by `vite.config.ts`; register it or fold the same logic into the canonical Timeline implementation before copying that behavior to Upscale.
- current `upscaleSelected()` queues immediately; extend the existing bulk-dialog state with `upscale` rather than creating a second modal.
- current standalone Director upscale still routes Topaz/KIE only; local Real-ESRGAN/FlashVSR/RIFE/GIMM choices must route to `/api/projects/v2/enhancer/jobs`, while existing Topaz choices remain on their external backend.
- migrate legacy `interpolateTo60Fps` UI/storage into VFI + target FPS + Smooth slow motion normalization without showing both systems at once.

### 37.10 Bulk Upscale semantics

`Upscale remaining`:

```text
eligible video base exists
AND no current valid upscale
AND no active upscale job
-> queue

current valid upscale -> skip
active upscale job    -> skip
still image           -> skipped_image
failed/cancelled job with no valid upscale -> eligible
```

`Re-upscale all`:

```text
eligible source/generated base exists
-> queue even when a current valid upscale exists
-> neural input is still the pre-upscale base according to Section 37.5
```

Never use the current upscale as an input merely because the action says "all including already-upscaled".

### 37.11 Release tests for this contract

Before rollout, integration tests must include one project containing all of these simultaneously:

```text
uploaded source video only
generated video only
uploaded + generated video on one segment
successful upscale from uploaded source
successful upscale from generated master
stale upscale after source replacement
failed replacement upscale with previous upscale retained
image segment mixed into the same Director timeline
```

Prove:

```text
Source selection feeds uploaded/source media
Generated selection feeds pre-upscale generated master
Upscaled selection feeds only the successful derivative
Re-upscale never sends upscaledVideoUrl to neural inference
Undo Upscale restores source/generated selection without deleting the upscale
Use Upscaled reactivates the stored derivative
TimelineRender receives exactly the selected per-segment URL
mixed uploaded/generated projects render correctly in one timeline
no existing D1 media field is renamed
no duplicate activeVideoUrl/currentVideoUrl field is introduced
```

### 37.12 Production Director row audit: current field behavior, not guessed naming

A real production export of `project_video_timeline` for `proj_1786739412370` was inspected before locking this contract. The row is `timelineVersion=video_director_v6`, revision 1507, with 91 segments. This is evidence for legacy/current compatibility; it is not a special-case project rule.

Observed media facts:

```text
91/91 segment.videoUrl values
  -> canonical H3 /video/previews/...-h264-preview.mp4

91/91 segment.renderVideoObjectKey values
  -> canonical H3 /video/generated/...-h265.mp4 master objects

64/91 segment.renderVideoUrl values
  -> H3 generated/master URL

27/91 segment.renderVideoUrl values
  -> H3 H.264 preview URL even though renderVideoObjectKey points at the H.265 master
```

Therefore legacy rows can be internally inconsistent in the URL fields while still retaining the correct master object key. Do not migrate or reinterpret the whole schema from field names alone.

The same production export also confirms:

```text
imageUrl
  -> actual current image asset when persisted

thumbnailUrl / videoThumbnailUrl / originalThumbnailUrl
  -> UI thumbnail/preview metadata only

sourceVideoUrl
  -> actual uploaded source video when one exists

originalImageUrl
  -> retained storyboard/original restore image, not necessarily the current selected image
```

A source resolver must use the actual populated media pointers plus current Director state. In particular, do not assume `mediaSource='video-uploaded'` proves that `sourceVideoUrl` exists on every historical row.

### 37.13 Provider-aware generated-media contract: H.264 is not synonymous with preview

The canonical concept is **authoritative original generated asset**, not "always H.265".

Provider behavior:

| Generator | Browser/editor generated media | Authoritative original generated media | Final TimelineRender / enhancer generated input |
|---|---|---|---|
| H3 | H.264 lightweight preview | H.265 master | H.265 master |
| Grok Imagine | H.264 result | the same H.264 result | the same H.264 result |
| Seedance | H.264 result | the same H.264 result | the same H.264 result |

Never classify a video as preview merely because its codec is H.264.

Preview detection is provider/contract-aware. For H3 V1 the canonical preview pattern is specifically:

```text
/video/previews/...-h264-preview.mp4
```

and the corresponding H3 generated master pattern is:

```text
/video/generated/...-h265.mp4
```

Arbitrary H.264 media from Grok, Seedance, uploaded sources, Topaz or future providers must not be rewritten to H.265 or rejected as a preview solely because it is H.264.

### 37.14 New-generation writeback must make future rows unambiguous

For new H3 generation completion, write both sides of the preview/master contract correctly:

```text
videoUrl             = H3 H.264 preview URL
videoObjectKey       = H3 H.264 preview object key
renderVideoUrl       = H3 H.265 master URL
renderVideoObjectKey = H3 H.265 master object key
```

For Grok Imagine and Seedance, there is only one generated result asset. It is valid and preferred for the same H.264 result to occupy both browser-playable and authoritative-render roles:

```text
videoUrl             = provider H.264 result URL
videoObjectKey       = provider H.264 result object key
renderVideoUrl       = SAME provider H.264 result URL
renderVideoObjectKey = SAME provider H.264 result object key
```

Do not manufacture a second H.265 asset just to make field names look symmetrical. `renderVideo*` means authoritative generated media for rendering/upscale; it does not mean "must be HEVC".

Future providers follow the same rule:

```text
one generated deliverable
  -> video* and renderVideo* may point at the same object

separate preview + master deliverables
  -> video* points at preview/browser asset
  -> renderVideo* points at authoritative original generated master
```

### 37.15 Legacy H3 preview rows: keep D1 as-is; resolve the master safely

Do **not** mass-edit the existing 27 audited H3 segments merely because `renderVideoUrl` contains the preview URL. They already retain the correct H.265 master identity in `renderVideoObjectKey`.

Canonical base-generated resolution for an H3/legacy segment is:

```text
1. valid renderVideoObjectKey
   -> resolve to the public/playable URL for that exact object

2. otherwise a renderVideoUrl that is NOT a canonical H3 preview
   -> use it

3. otherwise a canonical H3 preview URL/key
   -> derive /video/generated/...-h265.mp4
   -> verify that expected master object exists before use

4. provider single-result fallback
   -> use videoUrl only when it is not a canonical H3 preview, or when provider metadata says it is the authoritative single generated result
```

For the canonical H3 naming contract, this transformation is valid:

```text
/video/previews/NAME-h264-preview.mp4
        ↓
/video/generated/NAME-h265.mp4
```

So yes: for a confirmed H3 preview following that exact naming contract, replacing the preview path/suffix with the generated H.265 path/suffix points at the intended master **if that master object actually exists**. But this mapping belongs in the resolver as a fallback/compatibility rule, not as a blind global D1 string rewrite.

The existing production row already gives a stronger signal than string replacement because `renderVideoObjectKey` points at the master on all 91 audited segments. Prefer that object key first.

No one-time D1 migration is required before enhancer rollout. A later optional cleanup/backfill may normalize legacy `renderVideoUrl` values after the resolver is proven, but it is not required for correctness and must never be a prerequisite for Upscale All or TimelineRender.

### 37.16 Upscale All on current H3 projects must use master resolution before queueing

The old project may remain stored exactly as it is.

Before queueing any generated-video enhancer job:

```text
segment active/generated base
        ↓
baseGeneratedMediaForSegment()
        ↓
H3 legacy row with preview renderVideoUrl
        ↓
renderVideoObjectKey / verified canonical H3 master
        ↓
H.265 original generated master
        ↓
ESRGAN / FlashVSR / VFI
```

Therefore Upscale All on `proj_1786739412370` must send the H.265 H3 masters for the audited legacy/current H3 segments, including the 27 whose `renderVideoUrl` is stale/preview-shaped. It must not send `videoUrl` or the preview-shaped `renderVideoUrl` merely because they are truthy.

If the expected H3 master cannot be resolved or verified, do not silently upscale the H.264 preview. Mark that clip ineligible/failed with a clear master-missing error so the data problem is visible.

After successful upscale writeback, the old preview/master fields remain intact and `upscaledVideoUrl/upscaledVideoObjectKey` become the optional derivative. This means existing projects do not need destructive media-field repair just because they are being upscaled.

### 37.17 `pending_upscales` is short-lived operational evidence, not durable product truth

Completed/failed `pending_upscales` rows are expected to be deleted after roughly one hour. Therefore Section 17/37 references to exact `source_view`, source URL/object key and timing in `pending_upscales` mean **execution-time and short-term audit evidence only**.

Never depend on `pending_upscales` for long-term behavior such as:

```text
Undo Upscale days/weeks later
Re-upscale months later
which base variant produced the stored upscale
source-change staleness detection after job cleanup
TimelineRender media selection
```

Durable product truth stays in the existing Director segment JSON:

```text
activeVisualView
upscaledVideoUrl
upscaledVideoObjectKey
upscaledFromView
upscaledSourceUrl
upscaledSourceObjectKey
upscaledSourceTrimInMs
upscaledSourceTrimOutMs
upscaledSourceSpeed
upscaledTimingBaked
upscaledTargetFps
upscaledOutputDurationMs
```

`pending_upscales` duplicates the necessary source identity/timing while a job is alive so the backend can validate the no-recursive-upscale invariant and perform correct writeback. After retention cleanup, the segment remains sufficient by itself.

### 37.18 Provider/legacy release tests added by production audit

Add explicit tests for all of these generated-media shapes:

```text
new H3
  videoUrl = H.264 preview
  renderVideoUrl/ObjectKey = H.265 master
  -> editor may use preview
  -> enhancer + TimelineRender use H.265 master

legacy H3
  videoUrl = H.264 preview
  renderVideoUrl = H.264 preview
  renderVideoObjectKey = H.265 master
  -> no D1 migration required
  -> enhancer + TimelineRender still use H.265 master

Grok Imagine
  videoUrl/renderVideoUrl = same H.264 result
  -> editor + enhancer + TimelineRender all use that same authoritative H.264 file

Seedance
  videoUrl/renderVideoUrl = same H.264 result
  -> editor + enhancer + TimelineRender all use that same authoritative H.264 file

uploaded source video
  sourceVideoUrl = H.264 or other supported codec
  -> codec alone never makes it a preview
```

Assertions:

```text
H.264 codec alone never triggers H3 preview rewrite
only canonical H3 preview identity triggers preview->master compatibility mapping
renderVideoObjectKey wins over a preview-shaped legacy renderVideoUrl when it identifies the master
Upscale All never submits an H3 preview when the H3 master exists
Grok/Seedance H.264 originals are accepted as authoritative generated inputs
future H3 writes store preview and master fields cleanly
future Grok/Seedance writes may deliberately point video* and renderVideo* at the same H.264 object
```

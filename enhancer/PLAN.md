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

`CUDA 13.0 Update 2` is still CUDA **13.0**, not CUDA 13.1. RunPod/Novita scheduling should request/filter for CUDA 13.0-class hosts and then boot-qualify the actual driver/runtime. CUDA 13.x minor-version compatibility begins at NVIDIA driver 580, while CUDA 13.0 Update 2 is paired with driver 580.95.05; boot-time CUDA/CuPy/model/TensorRT smoke tests remain authoritative.

---

## 1. Product goal

Build a SceneBuilder image/video enhancer using RunPod and Novita GPU pods with the same operational discipline as H3 while preserving H3 implementation/state.

Product paths:

- Storyboard single-image upscale remains the existing legacy path unless deliberately changed later.
- Storyboard **Upscale All** uses local enhancer pods only; zero Replicate.
- Video Generation Timeline / Director video upscale.
- Local FAST and QUALITY enhancement.
- Local VFI with RIFE 4.9 or GIMM-VFI-F.
- Existing Topaz premium/external upscale/VFI choices remain behind their existing external backend.
- Video output targets: 1080p, 1440p/2K, 2160p/4K.
- VFI/output FPS product targets: **30, 48, 60 FPS only**.
- Source FPS may be arbitrary/variable: e.g. 16, 20, 23.976, 24, 25, 29.97, 30, 35, 48, 50, 59.94, 60, or VFR.
- H3-style priority, retries, heartbeats, error/debug state, idle reuse, configurable idle timeout, provision/job timeout, provider fallback, delete locks, delete verification/backoff, stale allocation cleanup, cancel and orphan cleanup.

Expensive spatial enhancement should normally create an enhanced master once. Trim/speed/FPS-only changes should reuse the valid enhanced master and rerun only timing/VFI when possible.

---

## 2. Hard non-negotiables

### 2.1 H3 lifecycle remains untouched

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

- Never insert enhancer workers into `h3_pod_workers`.
- H3 code never queries enhancer lifecycle tables.
- Enhancer never treats H3 workers as its pool.
- Enhancer has its own callback endpoint, HMAC derivation domain, replay state, delete locks and reaper state.
- Mature H3 runtime/lifecycle source behavior is not modified to make enhancer work.
- Stateless provider/R2 helper patterns may be reused only where they do not share lifecycle state.

### 2.2 Product D1/JSON contracts remain authoritative

Do not add enhancer-specific product schema to:

```text
projects_timeline
project_video_timeline
```

Do not create a second product-media state table. Detailed enhancer provenance belongs in operational enhancer rows.

### 2.3 Product R2 layout remains authoritative

Permanent outputs remain under existing media-layout ownership. Canonical semantic destinations:

```text
Storyboard image upscale:
projects/{projectId}/images/scenes/upscaled/

Director video upscale:
projects/{projectId}/video/upscaled/
```

No product-media R2 migration is part of enhancer work.

### 2.4 Replicate boundary

- `POST /api/upscale-batch` / Storyboard **Upscale All** -> local enhancer FAST pods only.
- New local jobs use enhancer-owned queue state such as `waiting_for_pod`, not legacy Replicate-claimable state.
- Existing single-image `/api/upscale` remains legacy Replicate unless deliberately changed later.

### 2.5 FILM is deleted

FILM is not installed, exposed, routed, benchmarked, documented as supported, or stored in either enhancer image.

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
TensorRT 10.14.1.48 runtime + builder
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
TensorRT 10.14.1.48 runtime for RIFE engine execution
FFmpeg with NVDEC/NVENC where available
NVML telemetry

Video upscale:
  FlashVSR v1.1 / qualified FlashVSR+ implementation path

VFI:
  RIFE 4.9
  GIMM-VFI-F
```

### 3.3 Topaz remains external

Keep the existing external premium contract for:

```text
Topaz Gaia 2
Topaz Proteus
Topaz Chronos where supported
Topaz Apollo where supported
```

Do not bake/redistribute Topaz assets in local enhancer Docker images or local TensorRT engine storage.

---

## 4. CUDA / PyTorch / CuPy / TensorRT lock

### 4.1 One CUDA toolkit/userspace line per Docker image

Both FAST and QUALITY use the same selected system CUDA line:

```text
CUDA Toolkit/userspace: 13.0 Update 2 (13.0.2)
```

Do not install CUDA 11, CUDA 12.x, or CUDA 13.1 beside it.

`PyTorch 2.13.0` is the framework version. `cu130` is its CUDA build/runtime tag, not a PyTorch major version.

Install one CuPy family only:

```text
cupy-cuda13x
```

Never let third-party installers add `cupy-cuda12x` alongside it.

### 4.2 TensorRT version

Pin:

```text
TensorRT 10.14.1.48
```

Reason: this is an NVIDIA-tested TensorRT 10.x release explicitly supporting CUDA 13.0 Update 2 and retaining the TensorRT 10 builder/runtime style required by our FP16 RIFE/ESRGAN path. Do not move to TensorRT 11.x until our own exporters are deliberately migrated and parity-tested.

### 4.3 Host/provider compatibility

Provider scheduling requests CUDA 13.0-class hosts where the API exposes CUDA selection.

Boot qualification is mandatory:

```text
nvidia-smi / driver inspection
CUDA allocation + real CUDA op
PyTorch CUDA availability/version
CuPy CUDA kernel
TensorRT load/build smoke where applicable
model-specific smoke tests
NVENC smoke where configured
```

A host that cannot execute the required CUDA 13.0 runtime correctly is never advertised ready; mark unhealthy, delete and reprovision. No CPU workaround.

---

## 5. GPU fleet policy

Supported architecture families:

```text
Ampere    -> primarily sm_86 fleet
Ada       -> sm_89
Blackwell -> sm_120 RTX/workstation
```

Global product limit:

```text
VRAM <= 48 GB, including 48 GB
```

Exclude A100/A800/H100/H200/B200 and other high-cost datacenter classes outside product economics even when technically usable.

Representative candidates subject to live inventory and qualification:

```text
Ampere: RTX 3090, RTX A4000/A4500/A5000, A40 48GB, RTX A6000 48GB
Ada:    RTX 4090, L4, L40/L40S 48GB, RTX 6000 Ada 48GB
Blackwell <=48GB: RTX 5090 and qualified workstation variants
```

QUALITY/FlashVSR starts at the approved RTX-4090-class floor or validated equal/higher allowed GPU.

---

## 6. GPU-only inference invariant

Neural inference may not silently use CPU.

```text
torch.cuda.is_available() == false -> worker unhealthy
model parameters/buffers unexpectedly on CPU at inference boundary -> fail/requeue
input tensors on CPU at neural boundary -> fail/requeue
framework/model offload to CPU -> capability/job failure
```

Explicitly own `CUDA_VISIBLE_DEVICES`, select `cuda:0`, move models/inputs there, validate outputs, and synchronize around diagnostics/benchmarks.

CPU is allowed for HTTP, JSON, R2 I/O, filesystem, ffprobe, orchestration, audio, codecs and controlled pre/post-processing. It is not a neural fallback.

### 6.1 Startup qualification

FAST before `/ready=true`:

```text
CUDA op
CuPy kernel
Storyboard ESRGAN native CUDA
video ESRGAN native CUDA
available video ESRGAN TRT deserialize/inference
RIFE native CUDA
available RIFE TRT deserialize/inference
GIMM-F native CUDA/CuPy
FFmpeg/NVENC smoke when configured
```

QUALITY:

```text
CUDA op
CuPy kernel
FlashVSR representative inference
RIFE native CUDA + available TRT
GIMM-F native CUDA/CuPy
FFmpeg/NVENC smoke
```

Advertise only capabilities that pass. Record GPU, VRAM, compute capability, driver, CUDA, PyTorch, CuPy, TensorRT, model capability and peak VRAM.

---

## 7. Exact TensorRT scope

TensorRT `.engine` files exist for exactly:

```text
1. realesr-animevideov3
2. realesr-general-x4v3
3. RIFE 4.9
```

Explicitly no TensorRT engine for:

```text
RealESRGAN_x4plus_anime_6B
RealESRGAN_x4plus
FlashVSR
GIMM-VFI-F
```

The two Storyboard image ESRGAN models are native PyTorch CUDA only.

### 7.1 Video ESRGAN rules

- FP16 TensorRT preferred when a valid engine matches.
- Static full-frame engines.
- No tiled ESRGAN TensorRT.
- No hidden ESRGAN tiling fallback.
- Missing/non-matching engine -> native full-frame PyTorch CUDA FP16.
- If full-frame PyTorch still OOMs after concurrency backoff -> requeue a larger allowed GPU or fail/retry; never CPU.

`realesr-general-x4v3` uses one fixed balanced denoise preset in V1; do not multiply engines for arbitrary DNI values.

### 7.2 BasicSR compatibility patch

When required by pinned upstream revision, only the deterministic import compatibility patch is approved:

```python
# old
from torchvision.transforms.functional_tensor import rgb_to_grayscale

# new
from torchvision.transforms.functional import rgb_to_grayscale
```

No architecture/weight/quality changes.

---

## 8. TensorRT engine topology

### 8.1 Video ESRGAN: 12 per compatibility set

Per model:

```text
Landscape:
854x480
1056x594
1280x720

Portrait:
480x854
594x1056
720x1280

= 6 static engines/model
```

Therefore:

```text
realesr-animevideov3 = 6
realesr-general-x4v3 = 6
video ESRGAN total   = 12
```

### 8.2 RIFE: exactly 3 engine files per compatibility set

```text
1080 class
  profile: 1920x1080
  profile: 1080x1920

1440 class
  profile: 2560x1440
  profile: 1440x2560

2160 class
  profile: 3840x2160
  profile: 2160x3840
```

Landscape + portrait are optimization profiles inside each resolution-class engine, not separate engine files.

```text
RIFE = 3 .engine files/set
```

### 8.3 RIFE TensorRT must keep `timestep` dynamic

Our RIFE ONNX/TensorRT adapter must expose runtime inputs equivalent to:

```text
img0
img1
timestep
```

`timestep` is a runtime value, not baked/constant-folded to `0.5`.

This is critical: the same spatial RIFE engine must accept arbitrary interpolation positions between two source frames. Do not build separate RIFE engines for x2/x3/x4, source FPS, target FPS or clip duration.

### 8.4 Engine count

Per compatibility target:

```text
12 video ESRGAN
 3 RIFE
----------------
15 engine files
```

Required baseline retained sets:

```text
sm_86 same-CC      = 15
sm_89 same-CC      = 15
sm_120 same-CC     = 15
AMPERE_PLUS        = 15
-----------------------
baseline cache     = 60 engines
```

Exact-GPU sets are optional and add another 15 only when deliberately generated/benchmarked.

### 8.5 Things that NEVER create another engine

```text
clip duration
total frame count
input FPS
30/48/60 target FPS
24->48
24->60
16->60
20->30
35->48
35->60
x1/x2/x3/x4 runtime concurrency
trim/range
Director playback speed
```

---

## 9. TensorRT portability and runtime lookup

Supported artifact modes:

```text
Exact GPU
SAME_COMPUTE_CAPABILITY sm_86
SAME_COMPUTE_CAPABILITY sm_89
SAME_COMPUTE_CAPABILITY sm_120
AMPERE_PLUS portable
```

Runtime preference:

```text
1. exact-GPU validated active engine
2. same-compute-capability validated active engine
3. validated AMPERE_PLUS engine
4. native GPU PyTorch fallback
```

Do not deserialize an engine whose recorded TRT/CUDA/model/hash/profile/hardware compatibility does not match runtime requirements.

---

## 10. TensorRT engine building

Yes: the CUDA 13.0 FAST image carries the full TensorRT 10.14.1.48 **builder + runtime**, so it can build the approved `.engine` files on actual RunPod/Novita GPUs.

Engine generation is an admin/offline control-plane job and is allowed to be slow. Build time is not customer inference latency. TensorRT tactic profiling may take minutes; that is acceptable because validated artifacts are generated once, uploaded to private R2 and reused.

Build on the intended compatibility target:

```text
sm_86 set      -> validated sm_86 GPU
sm_89 set      -> validated sm_89 GPU
sm_120 set     -> validated sm_120 GPU
AMPERE_PLUS    -> validated allowed builder configured for hardware compatibility
Exact GPU      -> exact requested allowed SKU
```

Use a compatible FAST pod; no separate builder Docker image is required. QUALITY consumes compatible RIFE engines built through this trusted path.

### 10.1 Build lifecycle

```text
admin Generate engine
-> pending_upscales job_type=engine_build
-> validate ALLOWED_EMAILS + server allowlists
-> resolve baked trusted ONNX/checkpoint hash
-> reuse/provision compatible FAST GPU pod
-> verify CUDA 13.0 + TRT 10.14.1.48 + GPU
-> build FP16 engine
-> deserialize/self-test
-> correctness comparison vs native CUDA reference
-> optional benchmark
-> SHA-256
-> upload to temporary private R2 key
-> callback
-> Worker verifies job/pod/object/hash/metadata
-> promote immutable final object
-> D1 transaction activates new engine
-> deactivate prior matching engine only after replacement succeeds
```

Failed replacement never disables the prior active engine.

---

## 11. TensorRT artifacts: Docker / R2 / D1

Docker owns durable sources:

```text
model code
weights/checkpoints
video-ESRGAN ONNX/export source
RIFE ONNX/export source
runtime dependencies
TRT builder/runtime
FFmpeg/NVML service code
```

Generated `.engine` binaries are never baked into Docker.

Private R2 root:

```text
models/.engine/
```

Allowed model trees only:

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

Use immutable names containing model/version/hash/TRT/CUDA/precision/profile/build identity. No mutable global `engine.plan`.

Example identity:

```text
...__trt-10.14.1.48__cuda-13.0.2__fp16__...
```

Temporary upload:

```text
models/.engine/.tmp/{engineBuildJobId}/...
```

D1 stores metadata/control state only, never binary bytes. `pending_upscales` remains the V1 job/history/engine-registry source of truth; do not create a parallel product-media state table. D1 records exact R2 key, SHA, size, model/version, ONNX hash, TRT/CUDA, compatibility, compute capability, profile, precision, builder GPU/provider, validation/benchmark data and active state.

Runtime does not scan R2 and guess. D1 selects the exact active engine; pod downloads/verifies it into disposable local cache and then deserializes it.

No provider network volume is required.

---

## 12. Video ESRGAN source-raster normalization

Known H3-origin enhancer inputs may map to canonical enhancer engines, but this does **not** put enhancer models into H3 pods.

Landscape mapping:

```text
.4 MP -> 854x480  -> exact
.6 MP -> 1056x594 -> exact
.7 MP -> 1138x640 -> may normalize to 1056x594 when quality rule permits
.9 MP -> 1280x720 -> exact
```

Portrait mirrors use portrait engines.

Never stretch. For small aspect mismatch, crop excess first, then Lanczos resize.

Example:

```text
1344x768
-> center-crop 1344x756
-> Lanczos 1280x720
-> static TRT engine
```

For non-canonical source where normalization would discard meaningful detail:

```text
native full-frame PyTorch CUDA FP16 at source raster
```

Never downscale 1920x1080 to 1280x720 solely to force an engine hit.

Neural x4 scale and final delivery resolution are separate:

```text
source -> x4 neural inference -> exact final resize/conform -> 1080/1440/2160
```

---

## 13. FPS/VFI contract: arbitrary source FPS, outputs only 30/48/60

Product output FPS values are exactly:

```text
30
48
60
```

Source rate is not restricted to those values.

### 13.1 CFR scheduling

For constant-frame-rate source FPS `S`, requested output FPS `T`, and effective playback-speed multiplier `p` when Director speed is intentionally baked:

```text
outputTime[n] = n / T
sourcePosition = outputTime[n] * S * p
left  = floor(sourcePosition)
right = left + 1
alpha = sourcePosition - left
```

For an output timestamp falling exactly on a source frame (`alpha == 0`), reuse that authoritative source frame. Otherwise interpolate between `left` and `right` using `alpha` as the timestep.

Examples with `p=1`:

```text
24 -> 48: source step = 24/48 = 0.5
24 -> 60: source step = 24/60 = 0.4
16 -> 60: source step = 16/60 = 0.266666...
20 -> 30: source step = 20/30 = 0.666666...
20 -> 48: source step = 20/48 = 0.416666...
35 -> 48: source step = 35/48 = 0.729166...
35 -> 60: source step = 35/60 = 0.583333...
```

Same model, same spatial engine. Only runtime `alpha/timestep` values and number of output timestamps change.

### 13.2 RIFE behavior

RIFE TensorRT and native adapters must accept arbitrary runtime timestep values. Do not reduce RIFE to integer multiplier-only logic.

For each required synthetic output frame:

```text
RIFE(frame[left], frame[right], timestep=alpha)
```

The RIFE `.engine` remains keyed by spatial resolution/profile + model/TRT/GPU compatibility, never by FPS ratio.

### 13.3 GIMM-VFI-F behavior

GIMM-VFI-F natively models motion continuously between adjacent frames and supports arbitrary interpolation timesteps.

For each required synthetic output frame:

```text
GIMM-F(frame[left], frame[right], timestep=alpha)
```

GIMM stays native PyTorch + CuPy CUDA; no TensorRT engine.

### 13.4 Target FPS lower than source FPS

If requested target FPS is below the authoritative source sampling rate, e.g.:

```text
35 -> 30
50 -> 48
60 -> 30
```

this is temporal downsampling/resampling, not an interpolation requirement. Do not waste RIFE/GIMM compute just to reduce FPS. Generate the requested target-time grid from source PTS and select/bracket source frames deterministically.

This does **not** violate the rule against duplicate/static-frame skipping: we are not detecting and deleting visually similar frames as an optimization; we are producing a deliberately lower requested output frame rate.

UI/backend should normalize an unnecessary local VFI selection to `None` for a pure downsample/pass-through case so the user is not charged for GIMM that did not execute.

### 13.5 Target FPS equal to source FPS

If target timing already equals source timing, do not invoke neural VFI.

### 13.6 VFR inputs

For variable-frame-rate sources, nominal FPS is not authoritative. Use `ffprobe`/presentation timestamps.

Create output timestamps on the exact 30/48/60 grid, find the bracketing source PTS for each output timestamp, and compute:

```text
alpha = (targetPTS - leftPTS) / (rightPTS - leftPTS)
```

Then reuse exact source frames or invoke RIFE/GIMM for `0 < alpha < 1` as required.

Preserve exact selected duration and audio/timeline semantics.

### 13.7 Scene-cut-aware VFI is mandatory

Detect or consume authoritative hard-cut boundaries. Never interpolate between the final frame before a hard cut and first frame after it.

At a cut:

```text
finish boundary timestamp according to deterministic policy
-> no synthetic cross-cut frame
-> restart VFI context in next scene
```

This applies to both RIFE and GIMM.

### 13.8 No duplicate/static-frame optimization

Do not add a detector that drops/collapses/substitutes duplicate-looking or static source frames to save compute. Preserve authoritative source timestamps. Disable third-party `static-skip`/duplicate-skip options unless a future explicit product decision changes this.

---

## 14. GIMM-VFI-F

Locked variant:

```text
GIMM-VFI-F
FlowFormer estimator
native PyTorch + CuPy CUDA
```

Do not substitute GIMM-VFI-R or `*-P` variants. No GIMM TensorRT.

Commercial production remains license-gated because upstream licensing is non-commercial by default unless commercial permission is obtained.

---

## 15. FlashVSR / QUALITY architecture

FlashVSR is QUALITY-only and native GPU.

```text
FlashVSR v1.1 / qualified FlashVSR+ lineage
no TensorRT
no CPU neural fallback
4090-class-or-better initial floor
```

SECourses Upscaler Pro is an implementation reference: standalone Python/Gradio, not a required ComfyUI dependency. Relevant architectural ideas we may reproduce from permissive/open upstreams after quality tests:

```text
2x/4x neural SR scale
optional pre-downscale to reach desired final raster
scene detection
scene/chunk processing
restartable chunk boundaries
GPU/output-dependent temporal chunk size
DiT tiling for VRAM control
large-tile/fewer-tile preference while staying below OOM/shared-memory spill
tile overlap adjustment if seams appear
VAE tiling disabled by default in full-model path
RIFE/GIMM as separate VFI stage
FFmpeg audio preservation
H.265/10-bit output
queue/cancel/health/progress
```

FlashVSR-specific DiT tiling is allowed. This does not change the separate no-tiling rule for video ESRGAN and RIFE.

Prefer full/non-DiT-tiled FlashVSR when validated VRAM allows; otherwise use the largest validated DiT tile that avoids OOM/shared-RAM spill. Do not use system shared memory as a hidden VRAM substitute.

Public Apache-2.0 FlashVSR+ implementations that replace Block-Sparse-Attention with Sparse SageAttention and support CUDA 13.0/Blackwell are candidates, not automatically trusted drop-ins. Compare against official FlashVSR v1.1 for fidelity and LCSA/attention behavior before production.

Qualify at minimum:

```text
sm_86
sm_89
sm_120
CUDA 13.0.2
PyTorch 2.13.0 cu130
4090/L40S/5090-class throughput + VRAM
DiT tiled vs full quality
scene-boundary continuity
exact frame count/duration/audio sync
```

---

## 16. Streaming video / codec / audio

Persistent path:

```text
authoritative input
-> ffprobe
-> decode/NVDEC where available
-> trim/range/timestamp preparation
-> approved aspect normalization
-> spatial upscale
-> optional scene-aware VFI
-> exact final resize/conform
-> H.265 encode/NVENC default
-> audio remux/timing handling
-> existing R2 upload
-> authenticated completion callback
```

Never implement video enhancement as thousands of JPEG/PNG dumps or one model process per frame. Keep model/engine persistent with bounded frame/task buffers.

Default enhanced master target:

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

Allowed-email admin encoder controls:

```text
NVENC CQ default 16, validated practical range
x265 CRF default 15, validated practical range
```

x265 CPU encoding is permitted; CPU neural inference is not.

If duration/timing is unchanged, preserve/remux original audio. VFI alone does not inherently change duration. If Director speed is intentionally baked into the VFI master, do not apply the same speed twice later.

---

## 17. D1 operational schema

### 17.1 `enhancer_pod_workers`

Store enhancer-only lifecycle/provider/GPU/runtime ownership including:

```text
worker id/service kind
provider/provider instance/endpoint
GPU class/name/compute capability/region
status/current job
runtime image + digest/build revision
CUDA/PyTorch/CuPy/TensorRT versions
capabilities + compact telemetry
heartbeat/error/debug
idle_since/idle_timeout/terminate_after
provider delete retry state
timestamps
```

### 17.2 Existing `pending_upscales`

Keep one physical unified job table and expand additively.

Job types:

```text
image_upscale
video_upscale
vfi_only
engine_build
engine_validate
benchmark
```

Persist project/scene/segment/payer identity, priority, requested/actual provider/GPU/worker, model/version/backend, engine metadata, input/settings/execution/output JSON, image source shape metadata, status/stage/progress, compact telemetry, attempts/errors/debug and timestamps.

Engine/history rows store exact R2 engine key/SHA/size, ONNX/checkpoint hash, TRT/CUDA, precision, compatibility target, compute capability/exact GPU, profile, builder identity, validation/benchmark and active state.

Keep separate:

```text
enhancer_pod_delete_locks
enhancer_event_nonces
enhancer_config
```

`enhancer_config` persists at least `idle_timeout_seconds` plus approved admin/runtime defaults.

---

## 18. Storyboard writeback and batching

Completion patches only existing product fields:

```text
scenes[n].image = new active upscaled image
storyboard.thumbnail = new active thumbnail when produced
preserve storyboard.originalImage
preserve storyboard.originalThumbnail
```

Do not use Storyboard prompt-enhancement `isEnhanced` as upscale provenance. Do not add nested enhancer project-state fields.

Keep previous successful image active until upload + authoritative patch succeeds.

Exact-shape image batching eligibility:

```text
same image model
same backend
same target
same exact input width
same exact input height
```

Default:

```text
same W×H -> up to x4
mixed W×H -> sequential on warm pod
OOM -> x4 -> x3 -> x2 -> x1
```

Do not resize/pad images just to form a batch.

---

## 19. Director writeback

Patch only:

```text
project_video_timeline.segments[n].upscaledVideoUrl
```

Detailed enhancer provenance stays operational.

Still-image rule:

```text
active authoritative media is video -> eligible
active authoritative media is image -> skipped_image
```

For `skipped_image`: no video job, no pod, no charge, no output and no media mutation. Bulk UI counts it separately from failure.

---

## 20. Provider scheduling and lifecycle

RunPod/Novita jobs are provider-neutral at the control plane.

RunPod creation mirrors the proven H3 operational shape: image, custom GPU priority/list, `gpuCount=1`, GPU compute, CUDA 13.0-compatible host selection, disk, env/ports, non-interruptible/non-locked policy.

Novita uses current equivalent fields for product/GPU count/rootfs/image/env/ports/network/billing and CUDA compatibility filtering where exposed.

Enhancer pod env contains only required enhancer worker token/control-plane URL/service kind/R2 access/idle timeout/debug/GPU-only flags/port.

Forced provider selection does not silently cross provider. `Auto` may fallback according to policy.

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

## 21. Runtime concurrency / OOM

Admin controls:

```text
Execution: Sequential | Parallel
Parallelism: x1 | x2 | x3 | x4 | Custom
Custom safety range initially 1..8
Auto CUDA OOM backoff: ON
```

One GPU = one active video job in V1. Video xN is internal work inside that one clip:

```text
video ESRGAN xN -> up to N same-size frame tasks in flight
RIFE xN         -> up to N pair/timestep tasks in flight
GIMM xN         -> only validated bounded concurrency
FlashVSR        -> conservative scene/chunk concurrency; x1 initially unless validated
```

OOM backoff:

```text
x4 -> x3 -> x2 -> x1
```

For custom, decrement safely. Persist known unsafe GPU/model/resolution/backend/concurrency combinations. If x1 full-frame still fails, requeue a larger allowed GPU. No ESRGAN/RIFE tiling and no neural CPU fallback.

---

## 22. Pod API / security

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

Stages include queued/waiting/downloading/probing/normalizing/upscaling/interpolating/encoding/uploading/verifying/completed/failed/cancelled.

Enhancer security:

- separate callback endpoint, e.g. `/api/projects/v2/enhancer/pod/events`;
- enhancer HMAC derivation domain `enhancer:${workerId}`;
- master secret may reuse `ENHANCER_POD_AUTH_MASTER_SECRET || H3_POD_AUTH_MASTER_SECRET`, but domain remains enhancer-specific;
- timestamp + nonce replay protection in enhancer state;
- timing-safe validation;
- callback bound to expected worker + job;
- signed/scoped R2 access;
- server-side allowlists for models/profiles/provider/GPU/precision/encoder settings;
- browser cannot send arbitrary ONNX URLs, `.engine` files or shell commands;
- admin APIs enforce existing SceneBuilder `ALLOWED_EMAILS` server-side.

---

## 23. Video Generation Timeline UI

Reuse existing `VideoGenerationTimeline` / Director surfaces. Do not create a separate normal-user enhancer page.

Keep/extend:

```text
Video/Image track:
  Upscale All
  Upscale Selected

Selected unit Controls:
  Upscale/Enhance
```

Batch modal shows eligible videos, already upscaled, remaining and skipped still images, with:

```text
Upscale remaining              # default
Upscale all including upscaled
```

Force reprocessing starts from authoritative pre-enhancement media, never recursively from the previous enhanced master. Keep previous successful master active until replacement succeeds.

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

Local VFI selector:

```text
None
RIFE 4.9
GIMM-VFI-F
```

Topaz exposes only compatible Topaz VFI choices.

Output FPS selector uses exactly:

```text
30
48
60
```

Backend accepts arbitrary source FPS. Do not model this as `interpolateTo60Fps: boolean`.

Conceptual fields:

```ts
interpolationModel?: 'none' | 'rife-4.9' | 'gimm-vfi-f' | 'topaz-chronos' | 'topaz-apollo';
targetFps?: 30 | 48 | 60;
applyDirectorPlaybackSpeed?: boolean;
```

If a selected output FPS is <= authoritative source sampling rate and no interpolation is required, normalize local VFI to None rather than charging for an unused model.

When VFI is used, expose `Apply Director playback speed`; do not double-apply speed.

Bulk status distinguishes queued/processing/completed/failed/skipped_image/cancelled.

---

## 24. Storyboard UI

Preserve:

```text
Upscale Image
Upscale All
Anime / Real
2K / 4K
batch progress
```

`Upscale All` uses local FAST only and zero Replicate.

Where scope is shown:

```text
Upscale remaining
Upscale all including already-upscaled
```

Including already-upscaled starts from committed/base pre-upscale image, not the current upscale.

Allowed-email Storyboard admin may expose provider/GPU, priority, idle timeout, xN, OOM backoff, pod telemetry/errors/debug. It must **not** show TensorRT build controls for image ESRGAN.

---

## 25. Video enhancer admin

Inside existing Video Generation Timeline Upscale/Enhance controls for `ALLOWED_EMAILS` only; API independently authorizes every action.

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
loaded model/engine/SHA
telemetry/heartbeat/error/debug
Stop/Delete pod
Manual dispatch
```

TensorRT builder model dropdown contains exactly:

```text
realesr-animevideov3
realesr-general-x4v3
RIFE 4.9
```

Never image ESRGAN, FlashVSR or GIMM.

Builder controls:

```text
trusted model/checkpoint/ONNX identity
FP16
compatibility: AMPERE_PLUS | sm86 | sm89 | sm120 | Exact GPU
provider Auto | RunPod | Novita
ESRGAN static shape or RIFE 1080/1440/2160 profile class
Generate
Validate
Benchmark
Force rebuild
Deactivate
Delete cached engine
Copy R2 key
View logs
```

Inventory shows model/profile/precision/compatibility/build GPU/provider/TRT/CUDA/file size/build duration/benchmark/timestamps/status.

Encoder controls expose NVENC CQ and x265 CRF. Benchmark UI may compare TRT vs native CUDA, providers, GPU classes, compatibility modes and NVENC vs x265; FlashVSR/GIMM may be benchmarked natively but never get TensorRT choices.

---

## 26. Local video pricing

Use existing SceneBuilder payer/team/credit/refund/idempotency mechanics.

Locked local video pricing:

```text
FAST + no VFI or RIFE 4.9 = 3 credits / billable second
FAST + GIMM-VFI-F         = 5 credits / billable second
QUALITY / FlashVSR        = 10 credits / billable second
```

QUALITY remains 10 credits/sec whether allowed local VFI is None, RIFE or GIMM unless explicitly changed later.

Rules:

- `skipped_image` = 0;
- retries/callback replays never double-charge;
- failure/cancel uses existing refund/reconciliation policy;
- Topaz keeps its existing premium/external billing path;
- billable duration is authoritative selected media range/duration actually being enhanced;
- if Director playback speed is baked, use resulting authoritative duration/range;
- use existing SceneBuilder credit rounding/normalization;
- if local VFI is bypassed because target FPS <= source and no neural interpolation executes, do not charge the FAST+GIMM premium merely because a stale UI selection existed.

---

## 27. Observability / errors

Pod telemetry includes provider/instance/GPU/compute capability/driver/CUDA/PyTorch/CuPy/TRT/VRAM/GPU utilization/memory utilization/temp/power/NVENC/NVDEC/CPU/RAM/current model/backend/engine/concurrency/dimensions/FPS/stage/progress/model FPS/end-to-end FPS/queue depth/elapsed/ETA when trustworthy.

Persist compact snapshots/summaries only; no one-second D1 time series.

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

Include bounded recent log tail and relevant GPU/model/engine/runtime/retry metadata.

---

## 28. Build architecture

Enhancer stays under top-level:

```text
enhancer/
  .dockerignore
  docker/
  src/
  models/
  scripts/
```

Do not move enhancer source into root H3 `src/` or enhancer Dockerfiles into root H3 `docker/`.

Layered conceptual DAG with no FILM:

```text
enhancer-smoke
-> enhancer-base (CUDA 13.0.2)
-> enhancer-torch (PyTorch 2.13.0 cu130, CuPy cuda13x, TRT 10.14.1.48)
   -> enhancer-vfi-models (RIFE + GIMM)
   -> enhancer-esrgan-models (4 ESRGAN; ONNX only for video TRT targets)
      -> enhancer-fast
   -> enhancer-flashvsr-runtime/models
      -> enhancer-quality
```

Model/checkpoint layers precede app/server code so server edits do not invalidate heavy layers. Remove package/build caches. Generated TensorRT engines are never Docker layers.

Keep H3 build workflow/root `scripts/remote_build.sh` behavior untouched. Enhancer uses its own Hetzner workflow and `enhancer/scripts/remote_build.sh` with enhancer-only build context.

Docker build != TensorRT build:

```text
Docker build -> temporary CPU/Hetzner builder -> runtime/model/ONNX image
TRT build    -> actual compatible NVIDIA GPU pod -> .engine -> private R2 + D1
```

Reuse existing repository secrets such as Docker Hub/Hetzner/HF/repo-clone credentials; never bake secrets into layers.

---

## 29. Repository rename

Approved target repository name:

```text
khuzaimamussawar/GPU-runtime
```

Rename is repository identity only. Do **not** rename H3 Docker Hub images, H3 D1 tables, task families, APIs or lifecycle identifiers.

Audit already established:

- maintained H3 Hetzner workflow passes dynamic `${GITHUB_REPOSITORY}`;
- H3 remote build clones `${GITHUB_REPOSITORY}` dynamically;
- `/opt/minimax-h3-serverless` is only a temporary local checkout-dir name and may remain initially;
- secret variable `GH_FH_TOKEN_MM_H3_SERVERLESS` may remain initially;
- enhancer build workflow also uses dynamic repository identity;
- SceneBuilder2 `minimax-h3-*` references are primarily Docker/runtime image/service identities, not a hardcoded GitHub clone dependency;
- no audited external `uses: khuzaimamussawar/minimax-h3-serverless/...@...` dependency was found.

After rename, update docs/developer git remotes. Do not rebuild H3 images solely because GitHub repository name changed.

---

## 30. Source/license policy

```text
Real-ESRGAN -> pinned permissive official upstream + deterministic BasicSR compatibility patch
RIFE        -> permissive upstream; build our own TRT adapters/exporters
FlashVSR    -> pin/review upstream + attention/runtime dependency licenses
GIMM-VFI-F  -> production blocked until commercial permission/license is confirmed
Topaz       -> external licensed backend; do not redistribute locally
```

Non-commercial community TensorRT wrappers may be used as technical references/benchmarks only; do not copy/vendor incompatible code into commercial runtime.

---

## 31. Qualification / release gates

Before production eligibility on each GPU/runtime class, verify:

```text
CUDA 13.0.2
PyTorch 2.13.0 cu130
CuPy CUDA kernel
TensorRT 10.14.1.48 builder/runtime as applicable
GPU-only invariant
native Storyboard ESRGAN
native video ESRGAN
video ESRGAN TRT build/load/parity
RIFE native + TRT with non-0.5 timestep values
GIMM-F arbitrary timestep CUDA/CuPy
FlashVSR quality/runtime on QUALITY GPUs
scene-cut behavior
no static-frame skip behavior
30/48/60 output scheduling from varied source FPS and VFR
NVDEC/NVENC
R2/auth/callback/idempotency
credit charge/refund behavior
```

Golden inputs include realistic/anime faces, hair/detail, text, foliage, gradients, camera pans, fast motion, occlusion, hard cuts, portrait/landscape, H3-origin rasters and slightly off-ratio inputs.

Measure inference/end-to-end FPS, VRAM, CPU/RAM, decode/encode, R2 transfer and output quality/parity.

---

## 32. Final canonical capability matrix

```text
FAST
  CUDA 13.0 Update 2 (13.0.2)
  PyTorch 2.13.0 cu130
  CuPy cupy-cuda13x
  TensorRT 10.14.1.48 builder + runtime
  FFmpeg NVDEC/NVENC

  IMAGE
    RealESRGAN_x4plus_anime_6B -> native GPU PyTorch only
    RealESRGAN_x4plus          -> native GPU PyTorch only

  VIDEO
    realesr-animevideov3       -> TRT FP16 preferred; native full-frame GPU fallback
    realesr-general-x4v3       -> TRT FP16 preferred; native full-frame GPU fallback

  VFI
    RIFE 4.9                   -> TRT FP16 preferred; dynamic runtime timestep; native GPU fallback
    GIMM-VFI-F                 -> native PyTorch + CuPy; arbitrary timestep; license-gated

QUALITY
  CUDA 13.0 Update 2 (13.0.2)
  PyTorch 2.13.0 cu130
  CuPy cupy-cuda13x
  TensorRT 10.14.1.48 runtime for RIFE
  FFmpeg NVDEC/NVENC

  VIDEO
    FlashVSR v1.1 / qualified FlashVSR+ -> native GPU only; no TensorRT

  VFI
    RIFE 4.9                   -> compatible TRT engine preferred
    GIMM-VFI-F                 -> native PyTorch + CuPy

OUTPUT FPS
  30 / 48 / 60 only
  arbitrary source FPS/VFR accepted

TRT ENGINE SCOPE
  realesr-animevideov3
  realesr-general-x4v3
  RIFE 4.9

TRT BASELINE CACHE
  15 sm86 + 15 sm89 + 15 sm120 + 15 AMPERE_PLUS = 60 engines
```

**No FILM. No Storyboard image ESRGAN TensorRT. No FlashVSR TensorRT. No GIMM TensorRT. No neural CPU fallback. No RIFE engine per FPS ratio.**

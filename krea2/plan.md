# SceneBuilder Krea 2 Runtime Plan

Status: **architecture / implementation plan**

Scope: add Krea 2 Turbo image generation to SceneBuilder Storyboard and CharacterScreen using the existing GPU-runtime + SceneBuilder2 control-plane patterns proven by H3 and Enhancer.

This file is the source of truth for the Krea 2 implementation. Runtime/provider proof comes first; SceneBuilder UI and D1 integration follow after the image canaries pass.

---

## 1. Locked product decisions

### Runtime/model

- Primary model: **Krea 2 Turbo INT8 ConvRot**.
- Runtime: **native ComfyUI Krea 2 support**; do not build a parallel Diffusers service for v1.
- Diffusion checkpoint: `krea2_turbo_int8_convrot.safetensors`.
- Text encoder: **Qwen3-VL-4B BF16** (`qwen3vl_4b_bf16.safetensors`).
- Qwen3-VL-4B is the native Krea 2 text encoder family in the pinned Comfy build and supports image inputs as well as text.
- Minimum GPU: **20 GB VRAM**.
- One generation at a time per GPU for v1.
- Providers: **RunPod Pods + Novita GPU instances**.
- Reuse H3/Enhancer auth, R2 and provider variables; do not create a third auth/config system.

### Storage

- **No network volume.**
- **No network disk.**
- **No provider-mounted model storage.**
- Runtime gets exactly **35 GB container/root filesystem disk**.
- Krea model, Qwen encoder and VAEs are baked into Docker layers.
- User/style LoRAs are **not baked into Docker**.
- LoRAs may be cached on the 35 GB container disk after first R2 download.
- Generated output may use normal temporary container files long enough to upload to project R2, then is cleaned.

Provider configuration:

```text
KREA2_POD_DISK_GB = 35
RunPod: containerDiskInGb = 35
Novita: rootfsSize = 35
Novita: networkStorages = []
RunPod: no network volume / volume mount
```

Because LoRA cache now legitimately uses local container disk, do **not** reserve the entire remaining space for an imaginary persistent-free policy. Instead the final baked image must leave enough space for:

```text
Comfy input/output/temp
one or more cached LoRAs
atomic R2 downloads
logs / small runtime scratch
```

Use a disk watermark/LRU policy rather than an absolute assumption that LoRAs never touch disk.

### VAE

User can select:

```text
qwen_image   -> qwen_image_vae.safetensors
wan_2_1      -> selected Wan 2.1 VAE checkpoint
```

Rules:

- `qwen_image` is default because it is the official/native Krea 2 path.
- `wan_2_1` is alternate and must pass matched-seed validation before normal production exposure.
- The Qwen3-VL text encoder is independent from the VAE selection.
- The same text conditioning can be used while choosing either supported VAE decode path.

### LoRAs

- **0 to 3 user-selected LoRAs per normal text-to-image generation.**
- Existing R2 objects stay exactly where they are:

```text
models/lora/krea2/MinimalisticVectorArtKrea2.safetensors
models/lora/krea2/Darkchurch_style_krea2_v1.0.safetensors
```

- Existing thumbnails are UI-only assets:

```text
models/lora/krea2/minimalist_vector_art_thumbnail.png
models/lora/krea2/dark_church_style_thumbnail.png
```

- GPU runtime never identifies a LoRA by display name.
- Every LoRA gets an immutable D1 `lora_id` (UUID/text primary key).
- Browser/API requests use `loraId` because two LoRAs may have the same visible name.
- Display names are explicitly **not unique**.
- R2 object key identifies the physical asset only after the Worker resolves a trusted D1 row.
- Optional SHA-256 may exist as integrity/debug metadata; it is not the identity or naming scheme.

---

## 2. Native ComfyUI Krea 2 baseline

Use the same tested Comfy revision currently used by H3 unless a Krea-specific canary proves a deliberate upgrade is required:

```text
COMFYUI_COMMIT = 2a68ce33b4c9ea6ee4283e618a74560cefb32694
H3 label: v0.31.0-9-g2a68ce33
comfy-kitchen = 0.2.28
```

That revision already contains:

- native `Krea2` model detection/model class;
- native `CLIPType.KREA2`;
- Krea-specific Qwen3-VL-4B hidden-state conditioning;
- Qwen3-VL image preprocessing;
- INT8/ConvRot quantization support through `comfy-kitchen`;
- CUDA 13 optimized quantized backend support.

Do not follow moving Comfy `master` in production.

---

## 3. CUDA / PyTorch / Python baseline

Start with the same proven software family as H3:

```text
CUDA: 13.0 / cu130
PyTorch: 2.13.0
Python: 3.13 target
Ubuntu: 24.04
```

Pinned Comfy quantization code enables the optimized `comfy-kitchen` CUDA path on CUDA 13+.

Final image must be measured against the real 35 GB provider rootfs. Build stages may use a CUDA devel image if required, but prefer a matching CUDA 13 runtime image for the final stage when the ConvRot canary proves it works.

Remove build-only caches/artifacts from final layers:

```text
apt lists
pip cache
HF cache
git history
compiler/build scratch
```

---

## 4. Docker layering — simple Krea chain, no H3 two-Comfy workaround

Krea does **not** need H3's historical `comfyui-artifact -> core overlay` workaround. H3 needed that because Comfy had already been hard-coded too early in the old lineage and later had to be overlaid after the heavyweight model chain.

For Krea, place Comfy in the correct position from day one:

```text
00 krea2-base
   CUDA 13 + Python + PyTorch 2.13/cu130 + system/runtime packages

10 krea2-model-int8
   FROM krea2-base
   add Krea 2 Turbo INT8 ConvRot

20 krea2-vaes
   FROM krea2-model-int8
   add Qwen Image VAE
   add selected Wan 2.1 VAE

30 krea2-qwen-bf16
   FROM krea2-vaes
   add Qwen3-VL-4B BF16

40 krea2-comfyui
   FROM krea2-qwen-bf16
   install/apply the exact pinned H3 Comfy commit
   install matching requirements incl. comfy-kitchen 0.2.28
   verify native Krea2 + KREA2 CLIP + ConvRot

50 krea2-nodes
   only custom/helper nodes actually needed by Krea/SceneBuilder

60 krea2-workflow
   baked API workflows + manifests only

70 krea2-runtime
   LAST layer
   pod HTTP server
   job contract
   R2 client
   workflow patching
   output upload
   LoRA cache manager
   memory cleanup/offload policy
   progress/cancellation/readiness/idle lifecycle
```

Why:

- changing Comfy does not redownload/rebuild Krea/Qwen/VAE parent layers;
- changing nodes rebuilds only nodes/workflow/runtime;
- changing workflow rebuilds workflow + runtime;
- changing runtime rebuilds only the final runtime layer;
- LoRA additions change D1/R2 only and rebuild nothing.

This intentionally follows the useful part of the current H3 pod ordering — nodes, then workflows, then the final pod/runtime layer — without copying H3's two-Comfy retrofit structure.

Final image target:

```text
khuxaima/scenebuilder-krea2-pod:latest
```

---

## 5. Hetzner layer-by-layer build

Add a dedicated workflow:

```text
.github/workflows/hetzner-krea2-build.yml
krea2/scripts/remote_build.sh
```

Mirror H3/Enhancer temporary-builder behavior:

```text
target
targets_csv
image_tag
server_type
location
hetzner_image
debug_keep_server_on_failure
delete_server_after_success
debug_keep_minutes
docker_build_attempts
```

Recommended targets:

```text
base
model-int8
vaes
qwen-bf16
comfyui
nodes
workflow
runtime
```

Push every successful parent to Docker Hub so later attempts reuse the heavyweight published parents.

---

## 6. Resolution contract

### What Krea 2 actually supports

Krea 2 is not locked to one fixed image shape. The official open inference path accepts variable width/height in the 1K-to-2K operating range and pads dimensions to a multiple of 16 when needed. Native Comfy also exposes Krea 2 through a megapixel/aspect-ratio selector rather than a fixed 1024x1024-only graph.

For SceneBuilder Storyboard, exact 16:9 / 9:16 and 16-pixel alignment are more important than reproducing Comfy's approximate 1.0 MP number exactly.

### Recommended SceneBuilder presets

Use:

```text
1K landscape: 1280 x 720
1K portrait:   720 x 1280

2K landscape: 2048 x 1152
2K portrait:  1152 x 2048
```

Reasons:

- all four are exact 16:9 / 9:16;
- all four dimensions are divisible by 16, matching Krea's native padding/alignment expectation;
- `1280x720` is ~0.92 MP and is a clean practical 1K-class Storyboard frame;
- `2048x1152` is the clean true-2K-long-edge 16:9 preset;
- no hidden 8/16-pixel pad is needed just to satisfy the model;
- no post-generation crop is needed to restore Storyboard framing.

### Why not 1368x768 as the product default

Comfy's built-in megapixel selector calculates approximately `1368x768` for 16:9 around its 1.0 MP target because that utility rounds to its configured grid. That is a calculator artifact, not a special Krea training resolution.

`1368x768` is also not an exact 16:9 16-pixel-aligned pair: 1368 is not divisible by 16. Krea can pad it internally, but SceneBuilder gains nothing from that hidden padding.

Therefore:

```text
normal SceneBuilder 1K default = 1280x720 / 720x1280
```

Canary both `1280x720` and the Comfy-style ~1 MP shape once for quality comparison. If an actual quality regression appears at 1280x720, reconsider `1536x864` rather than adopting a slightly-off-ratio padded size.

### Optional 2K performance preset

If `2048x1152` is too slow/OOM-prone on qualified 20 GB cards, an intermediate exact 16:9 aligned preset can be exposed later:

```text
1792 x 1008
1008 x 1792
```

Do not silently replace the requested 2K tier; route/escalate according to GPU policy.

---

## 7. Sampling/settings contract — user controls generation

Native Turbo defaults:

```text
steps:     8
cfg:       1.0
sampler:   euler
scheduler: simple
denoise:   1.0
seed:      random unless pinned
VAE:       qwen_image
```

These are defaults, not hidden constants. Expose:

```text
prompt
prompt enhancement on/off
negative prompt
resolution tier/orientation
seed / random seed
steps
CFG
sampler
scheduler
denoise
VAE
0-3 user LoRAs OR style-reference images
strength for every selected LoRA
```

Initial advanced ranges:

```text
steps:       1-50, default 8
cfg:         configurable, default 1.0
sampler:     tested allowlist; Euler guaranteed
scheduler:   tested allowlist; simple guaranteed
denoise:     0.0-1.0, default 1.0
seed:        explicit integer or random
VAE:         qwen_image | wan_2_1 after Wan canary
```

### Negative prompt

Native Turbo default uses zeroed negative conditioning. At `CFG=1`, a text negative prompt is effectively inactive.

SceneBuilder can expose negative prompt as advanced:

- CFG 1 -> show it as inactive/no effect;
- CFG > 1 -> encode supplied negative text and use it as guided negative conditioning;
- benchmark this separately because Turbo is distilled around the native low/no-guidance behavior.

### Prompt enhancement

Always persist the raw user prompt. If enhancement is enabled, also persist the effective/enhanced prompt in execution metadata. Regeneration must never lose the original prompt.

---

## 8. Two mutually exclusive style paths in v1

SceneBuilder v1 uses one of two style modes per generation:

```text
A. user LoRA mode
   1-3 user-selected D1 LoRA IDs
   no style-reference images

B. style-reference mode
   no user-selected LoRA
   1-3 style-reference images
   runtime uses the dedicated internal Krea style-reference adapter/workflow
```

This matches the product rule: **if there is no user LoRA, style images may be sent instead.**

Do not combine arbitrary user LoRAs + style-reference images in v1; add that only after a dedicated memory/quality canary.

### Internal style-reference adapter

The official/native Krea style-reference workflow uses `krea2_style_reference.safetensors` internally. Treat it as a hidden/system LoRA:

- not visible as a user-selectable style LoRA;
- not baked into Docker;
- registered as an internal D1 `lora` row or trusted runtime asset;
- downloaded from R2 and cached on container disk like other LoRAs;
- automatically applied by the style-reference workflow;
- does not count as one of the user's 0-3 visible LoRA selections because style-reference mode requires zero user LoRAs.

---

## 9. Style-reference image dimensions

Style images do **not** need to match:

```text
each other
output resolution
output aspect ratio
```

Do not stretch/crop every reference to `1280x720` or `2048x1152` just because that is the output frame.

Native Qwen3-VL image preprocessing handles every image independently. In the pinned Comfy implementation the Krea Qwen3-VL path uses patch size 16 and the generic Qwen vision preprocessing rounds each image independently to its required grid while approximately preserving its source aspect ratio. Native Krea also processes reference latents individually before concatenating their tokens.

Runtime policy:

```text
1. EXIF-orient
2. convert to RGB
3. preserve source aspect ratio
4. do not force output aspect ratio
5. only downscale if source is unreasonably large for our 20 GB policy
6. let the pinned native Krea/Comfy workflow perform its own model alignment
```

Initial safety cap for very large uploaded references should be canary-driven. Start by testing 1-3 mixed references such as:

```text
square + landscape + portrait
1024x1024 + 1280x720 + 720x1280
odd user dimensions that are not multiples of 16/32
```

The acceptance condition is that mixed-size references work without pre-cropping and without avoidable OOM.

---

## 10. Generic D1 LoRA catalog

Create one generic catalog for future image and video LoRAs.

### `lora`

```sql
CREATE TABLE IF NOT EXISTS lora (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    description TEXT,

    lora_type TEXT NOT NULL DEFAULT 'image',
    category TEXT,

    r2_object_key TEXT NOT NULL UNIQUE,
    thumbnail_object_key TEXT,
    file_name TEXT,
    file_size_bytes INTEGER,
    sha256 TEXT,

    trigger_words_json TEXT NOT NULL DEFAULT '[]',
    tags_json TEXT NOT NULL DEFAULT '[]',

    enabled INTEGER NOT NULL DEFAULT 1,
    visibility TEXT NOT NULL DEFAULT 'private',
    is_internal INTEGER NOT NULL DEFAULT 0,

    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
```

Notes:

- `id` is immutable generated UUID/text and is what UI/API jobs persist.
- `display_name` is **not unique**.
- two LoRAs may have the same visible name.
- `r2_object_key` identifies one physical stored asset.
- `thumbnail_object_key` is UI-only.
- `sha256` is optional integrity metadata, not identity.
- `lora_type`: `image | video | both`.
- `is_internal=1` hides system adapters such as Krea style-reference from normal UI catalogs.

### `lora_model_support`

```sql
CREATE TABLE IF NOT EXISTS lora_model_support (
    lora_id TEXT NOT NULL,
    model_key TEXT NOT NULL,
    quantization TEXT NOT NULL DEFAULT '*',

    min_strength REAL,
    max_strength REAL,
    default_strength REAL,

    min_steps INTEGER,
    max_steps INTEGER,
    recommended_steps INTEGER,

    apply_target TEXT NOT NULL DEFAULT 'model_only',
    notes TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,

    PRIMARY KEY (lora_id, model_key, quantization),
    FOREIGN KEY (lora_id) REFERENCES lora(id)
);
```

This allows one physical R2 asset to support multiple models/quantizations without duplication.

Seed the two existing user LoRAs with generated immutable IDs. Also register the Krea style-reference adapter as hidden/internal if style-reference mode is enabled.

---

## 11. LoRA disk + RAM cache policy

The earlier RAM-only assumption is removed.

### Disk cache

Use container disk as a bounded local LoRA cache:

```text
/opt/scenebuilder-krea2/cache/loras/<lora_id>.safetensors
```

Use D1 `lora_id` for cache identity, not display name or original filename, so same-name LoRAs cannot collide.

Flow on first use:

```text
browser loraId
 -> Worker resolves D1 row/support
 -> pod gets trusted loraId + R2 object key
 -> if valid cached file exists: reuse
 -> otherwise R2 GetObject to temporary .part file
 -> verify expected size and optional sha256 when available
 -> atomic rename to <lora_id>.safetensors
 -> load tensors/state dict
```

The browser never supplies a filesystem path or arbitrary R2 key.

### Hot RAM cache

Keep the currently useful LoRA tensor/state dict in CPU RAM when a job completes if memory pressure is acceptable.

Key hot cache by the immutable asset identity/version, e.g.:

```text
lora_id + object version/etag or sha256 when known
```

Do **not** key by strength. The same parsed LoRA tensors can be reapplied at a different strength on the next job.

If the next job uses the same LoRA set:

```text
reuse hot CPU tensors
avoid rereading from disk/R2
repatch at requested strengths
```

If the next job uses different LoRAs:

```text
release old LoRA tensors from RAM as needed
leave their safetensors in bounded disk cache
load new LoRA from disk cache or R2
```

### Disk eviction

Because root disk is only 35 GB, use disk watermarks/LRU:

```text
never evict baked model/encoder/VAE assets
never evict an active job's LoRA
prefer deleting oldest unused LoRA cache files first
redownload from R2 later when needed
```

Exact cache budget is set after final-image size is measured. It must be dynamic from actual free disk, not a hard-coded fantasy capacity.

---

## 12. Model/encoder/VAE warm-memory policy

The expensive **Krea model and Qwen text encoder are what we mean by CPU offload**.

Goal after a completed job:

```text
VRAM: job-specific allocations cleared / mostly idle
CPU RAM: keep reusable Krea model + Qwen encoder + selected VAE/model objects where safe
Disk: baked checkpoints remain; LoRA safetensors cache remains
```

Do not reload 13+ GB Krea and ~9 GB Qwen from container storage for every job if the pod remains warm.

### Job sequence on constrained GPU

```text
1. obtain Qwen from warm CPU state or load once from baked disk
2. move required encoder portions to GPU
3. encode prompt and optional style-reference image conditioning
4. move/offload Qwen back to CPU RAM
5. clear reclaimable CUDA allocator cache
6. obtain Krea from warm CPU state or load once from baked disk
7. move/stream required Krea weights to GPU under Comfy memory manager
8. obtain user/internal LoRAs from hot RAM or local disk/R2 cache
9. patch Krea model
10. sample
11. activate selected VAE, decode
12. upload result
13. remove job temp input/output
14. retain reusable model objects/weights in CPU RAM
15. retain useful hot LoRA tensors in CPU RAM if within cap
16. clear job-specific GPU allocations/cache
```

### Comfy cleanup semantics

Do **not** use a cleanup action equivalent to `unload_models=true` after every job because that defeats warm reuse and can unload model state from RAM.

Desired idle cleanup is equivalent to:

```text
free cached/reclaimable GPU memory
keep reusable model objects/offload state
```

Use the pinned Comfy memory manager and, where appropriate, `/free` with `free_memory=true` but **without** `unload_models=true`. Canary the exact behavior on the pinned revision and add a small helper if we need an explicit GPU->CPU offload while preserving warm host state.

On real host-memory pressure, evict in this order:

```text
old hot LoRA tensors
inactive VAE state
text encoder warm state
Krea warm state last
```

All baked assets remain available on container disk for reload.

---

## 13. Enhancer engine pattern to copy

Enhancer's TensorRT runtime already demonstrates the warm-worker pattern we want conceptually:

```text
_ENGINE_CACHE
  keeps deserialized TensorRT engine objects across jobs

_EXECUTION_CACHE
  keeps job/use execution context + CUDA stream
```

At job end Enhancer releases the execution contexts/CuPy pool and calls `torch.cuda.empty_cache()`, but it does **not** evict the process-level `_ENGINE_CACHE`. The next job therefore avoids downloading/deserializing the same engine again while job-specific GPU workspace is reclaimed.

Krea analogue:

```text
persistent/warm CPU cache:
  Krea model object/weights
  Qwen model object/weights
  VAE objects
  current LoRA tensor states

job-specific GPU state:
  activations
  execution tensors
  sampler state
  temporary patches/workspaces
```

At job completion, reclaim the second category while preserving the first as host RAM allows.

---

## 14. Native workflows / manifests

Use two explicit API workflows rather than dynamically mutating one giant graph:

```text
krea2/workflows/krea2_turbo.json
krea2/workflows/krea2_style_reference.json

krea2/workflows/manifests/krea2_turbo.json
krea2/workflows/manifests/krea2_style_reference.json
```

### Normal text/LoRA workflow

```text
UNETLoader
  krea2_turbo_int8_convrot.safetensors

CLIPLoader
  qwen3vl_4b_bf16.safetensors
  type = krea2

SceneBuilder MODEL LoRA patch chain (0-3)
CLIPTextEncode positive
optional guided negative encode, otherwise ConditioningZeroOut
EmptyLatentImage
KSampler
VAELoader selected VAE
VAEDecode
SaveImage / runtime output
```

### Style-reference workflow

Start from the official/native Krea style-reference graph behavior:

```text
1-3 style images
internal krea2_style_reference adapter
Krea INT8 model
Qwen3-VL-4B BF16
selected VAE
prompt / optional prompt enhancement
seed / width / height
native reference latent conditioning
sampling / decode / output
```

Do not rewrite the reference algorithm before reproducing the official workflow successfully.

---

## 15. Provider-neutral payload

Normal LoRA example:

```json
{
  "jobId": "job_123",
  "projectId": "proj_123",
  "taskFamily": "krea2_image",
  "model": "krea-2-turbo",
  "prompt": "...",
  "negativePrompt": "",
  "width": 1280,
  "height": 720,
  "settings": {
    "styleMode": "lora",
    "resolutionTier": "1k",
    "seed": 12345,
    "steps": 8,
    "cfg": 1.0,
    "sampler": "euler",
    "scheduler": "simple",
    "denoise": 1.0,
    "promptEnhance": false,
    "textEncoder": "qwen3vl_4b_bf16",
    "vae": "qwen_image",
    "loras": [
      { "loraId": "7c5d...uuid...", "strength": 0.8 }
    ]
  },
  "inputs": {
    "outputPrefix": "projects/proj_123/images/generated"
  }
}
```

Style-reference example:

```json
{
  "jobId": "job_124",
  "projectId": "proj_123",
  "taskFamily": "krea2_image",
  "model": "krea-2-turbo",
  "prompt": "...",
  "width": 1280,
  "height": 720,
  "settings": {
    "styleMode": "reference_images",
    "seed": 12345,
    "steps": 8,
    "cfg": 1.0,
    "sampler": "euler",
    "scheduler": "simple",
    "denoise": 1.0,
    "vae": "qwen_image",
    "loras": []
  },
  "inputs": {
    "styleImages": [
      { "objectKey": "projects/proj_123/style/a.png" },
      { "objectKey": "projects/proj_123/style/b.jpg" }
    ],
    "outputPrefix": "projects/proj_123/images/generated"
  }
}
```

Worker validates `styleMode`: user LoRAs and style images are mutually exclusive in v1.

---

## 16. Attention implementation

Do **not** add SageAttention, FlashAttention or special Krea attention nodes in v1.

Native Krea 2 uses Comfy's core optimized attention path. Start with:

```text
PyTorch 2.13/cu130
pinned Comfy
comfy-kitchen INT8 ConvRot
native Comfy attention selection
```

Only add an alternate backend after controlled benchmarks on actual RunPod/Novita GPU classes prove a benefit.

---

## 17. GPU policy

Reuse the Enhancer GPU catalog and nominal-VRAM tolerance.

```text
KREA2_MIN_VRAM_GB = 20
```

Eligible RunPod classes begin with existing >=20 GB catalog entries such as RTX 4000 Ada, RTX A4500, L4, 4090, A5000, 3090, RTX PRO 4000, 5090, RTX PRO 4500 variants, L40/L40S, RTX 6000 Ada, A40 and A6000.

Current Novita candidates are 4090, 5090, RTX 6000 Ada and L40S.

Every GPU class requires a Krea canary because INT8 ConvRot behavior varies by architecture.

OOM escalation:

```text
20 GB -> 24 GB -> 32 GB -> 48 GB
```

Do not repeatedly retry the same failed tier.

Host RAM matters because warm CPU offload is intentional. Initial qualification target remains >=32 GB host RAM, with larger RAM preferred when available.

---

## 18. RunPod / Novita lifecycle

Follow H3/Enhancer GPU-instance provisioning rather than older endpoint-per-GPU serverless routing.

Reuse:

```text
RUNPOD_API_KEY
NOVITA_API_KEY
R2_BUCKET_NAME
R2_ENDPOINT
R2_ACCESS_KEY
R2_SECRET_KEY
R2_REGION
R2_PUBLIC_URL
H3_POD_AUTH_MASTER_SECRET
```

Per worker:

```text
SCENEBUILDER_POD_TOKEN = HMAC(H3_POD_AUTH_MASTER_SECRET, workerId)
```

Pod API:

```text
GET  /health
GET  /ready
POST /workloads
GET  /workloads/:id
POST /workloads/:id/cancel
POST /workloads/:id/items/:jobId/cancel
```

Progress phases:

```text
queued
allocating
cold_start
preparing_model
loading_loras
loading_style_references
encoding_prompt
generating
decoding
uploading
completed
failed
cancelled
```

One active job per GPU for v1. Warm workers accept later Krea jobs until idle expiry.

---

## 19. SceneBuilder2 integration

Reuse the current image-generation path:

```text
CharacterScreen / Storyboard
  -> apiGenerateImage()
  -> /api/generate-image
  -> image_generation_jobs
  -> RunPod/Novita Krea pod dispatch
  -> existing status polling
  -> existing cancellation
  -> R2 finalization
  -> existing thumbnail flow
```

Do not create a second browser polling protocol.

Persist an immutable settings snapshot containing:

```text
model key + quantization
resolution tier + width/height
raw prompt
promptEnhance flag
effective/enhanced prompt
negative prompt
seed
steps/cfg/sampler/scheduler/denoise
VAE
text encoder profile
styleMode
ordered user loraId + strength list OR style image object keys
requested compute backend
```

Shared UI LoRA component:

```text
<LoraSelector modelKey="krea-2-turbo" maxSelected={3} />
```

When at least one user LoRA is selected, style-image controls are disabled for v1. When zero user LoRAs are selected, the user may choose/upload 1-3 style images.

---

## 20. D1/catalog API behavior

Conceptual query:

```text
GET /api/loras?model=krea-2-turbo&quantization=int8_convrot&type=image
```

Return normal user-visible rows only:

```text
id
displayName
description
thumbnailUrl
triggerWords
minStrength
maxStrength
defaultStrength
minSteps
maxSteps
recommendedSteps
```

Internal adapters are excluded from normal catalog response.

Worker resolves `loraId` again at dispatch time so stale/forged browser data cannot bypass compatibility rules.

---

## 21. Tests / canaries

### Build/runtime

```text
35 GB rootfs policy respected
no network volume configured
CUDA 13 visible
PyTorch 2.13 + cu130
pinned H3 Comfy commit installed after heavy Krea/Qwen/VAE layers
comfy-kitchen 0.2.28 present
native Krea2 recognized
Qwen3-VL-4B KREA2 CLIP recognized
ConvRot backend works
no SageAttention/FlashAttention dependency
```

### Resolution matrix

Matched prompt/seed/settings:

```text
1280x720
720x1280
1368x768 comparison only
2048x1152
1152x2048
1792x1008 fallback/performance comparison
```

Record quality, VRAM, RAM, generation time and hidden padding behavior.

### Style-reference matrix

```text
1 style image
2 style images
3 style images
square + landscape + portrait in same request
mixed dimensions
odd non-aligned dimensions
zero user LoRAs enforced
internal style-reference adapter auto-loaded
```

Verify references are not stretched/cropped to output aspect ratio unless the native workflow itself requires it.

### VAE matrix

```text
Qwen Image VAE
Wan 2.1 VAE
```

Same prompt/seed/resolution. Compare correctness, color, detail, artifacts, decode time and memory.

### LoRA cache matrix

```text
0 LoRA
MinimalisticVectorArt only
Darkchurch only
both together
3-LoRA test when available
same display name on two different D1 IDs -> correct asset selected
invalid ID rejected
4 user LoRAs rejected
first R2 download -> disk cache
second same-LoRA job -> hot RAM reuse when available
same asset different strength -> reuse tensors, repatch strength
new LoRA -> old hot tensors evictable, disk cache retained
cache watermark/LRU deletes oldest unused file when needed
```

### Warm-memory lifecycle

After every job record:

```text
VRAM allocated/reserved
host RAM
which model objects remain warm
which LoRA states remain hot
local disk cache bytes
```

Verify:

```text
job-specific GPU memory is released
Krea/Qwen do not reload from disk unnecessarily on warm next job
same LoRA can reuse CPU tensors
new LoRA can replace hot tensors without losing disk cache
memory pressure can evict warm states safely
```

### Provider matrix

At minimum:

```text
RunPod >=20 GB canary
RunPod 4090
RunPod 5090
Novita 4090
Novita 5090
48 GB fallback class
```

### Lifecycle

```text
queued cancel
active Comfy interrupt
pod crash
callback retry/loss
R2 LoRA read failure
R2 output upload failure
20 -> 24 -> 32 -> 48 OOM escalation
warm reuse
idle deletion
GPU cleanup without destroying useful host-RAM warm cache
```

---

## 22. Rollout order

### Phase 1 — base runtime

1. Build CUDA 13/PyTorch 2.13 base.
2. Build Krea INT8 -> both VAEs -> BF16 Qwen heavy chain.
3. Add the pinned H3 Comfy revision **after** the heavy model layers.
4. Add nodes -> workflow -> final runtime.
5. Verify final image fits 35 GB provider disk with usable LoRA/temp headroom.
6. Reproduce native Turbo defaults.
7. Canary `1280x720`, `720x1280`, `2048x1152`, `1152x2048`.
8. Pass 20 GB worker canary with CPU offload.

### Phase 2 — warm memory lifecycle

1. Keep Krea/Qwen/VAE model objects warm in CPU RAM after jobs.
2. Clear job GPU allocations/cache without full model unload.
3. Validate second-job warm latency and host RAM.
4. Add pressure-aware CPU eviction policy.

### Phase 3 — D1 + LoRA cache

1. Create generic `lora` + `lora_model_support` schema.
2. Seed two existing R2 LoRAs with immutable IDs.
3. Implement trusted R2 -> local disk cache.
4. Implement hot CPU tensor cache for current LoRAs.
5. Add dynamic disk watermark/LRU.
6. Test 0-3 user LoRAs and same-name collision.

### Phase 4 — style-reference mode

1. Register/internalize `krea2_style_reference` adapter outside normal user catalog.
2. Reproduce official native style-reference workflow.
3. Support 1-3 mixed-size style images when zero user LoRAs are selected.
4. Validate no forced output-resolution resize is needed.

### Phase 5 — provider pods

1. Add RunPod/Novita lifecycle.
2. Hard-set provider disk to 35 GB and no network storage.
3. Reuse H3 auth/provider/R2 vars.
4. Add readiness, warm reuse, OOM escalation and deletion.

### Phase 6 — SceneBuilder UI/control plane

1. Add `krea-2-turbo` to existing image jobs.
2. Preserve async polling/cancel/finalization APIs.
3. Add LoRA catalog by immutable ID.
4. Add LoRA-or-style-images mutually exclusive controls.
5. Add advanced generation controls.

---

## 23. Guardrails

- No network volume/network disk.
- Container/root disk is 35 GB.
- LoRAs are not baked into Docker.
- Local LoRA disk cache is allowed and bounded by LRU/watermarks.
- Never identify LoRAs by display name.
- D1 immutable `lora_id` is API/job/cache identity.
- Never accept arbitrary browser R2 object keys for LoRA selection.
- Maximum 3 user-selected LoRAs.
- Style-reference mode uses zero user LoRAs in v1.
- Style images may have different dimensions/aspects; do not force them to output dimensions.
- Keep Krea/Qwen warm in CPU RAM where host memory allows.
- Clear job GPU state without blindly unloading all model RAM state.
- Native Turbo defaults remain 8 / CFG1 / Euler / simple / denoise1.
- No SageAttention or FlashAttention dependency in v1.
- Use the same pinned Comfy commit as H3 unless a Krea canary forces a deliberate upgrade.
- Krea layering is simple: heavy models -> Comfy -> nodes -> workflow -> runtime.
- No separate browser polling system for Krea.
- No separate Krea auth master secret.

---

## 24. First implementation milestone

```text
35 GB container disk only
no network storage
CUDA 13 / PyTorch 2.13 cu130
Krea 2 Turbo INT8 ConvRot
Qwen3-VL-4B BF16
Qwen Image VAE + Wan 2.1 VAE baked
pinned H3 Comfy installed after heavy layers
nodes -> workflow -> runtime
no Sage/FlashAttention
1280x720 / 720x1280
2048x1152 / 1152x2048
8 steps / CFG1 / Euler / simple / denoise1 defaults
one RunPod >=20 GB canary
warm CPU offload lifecycle proven
```

Acceptance criteria:

```text
Comfy boots
native Krea2 + KREA2 Qwen CLIP load
BF16 encoder works
Krea/Qwen can move between GPU and CPU/offload state on 20 GB worker
job GPU allocations are cleared after completion
warm second job avoids unnecessary checkpoint reload
both VAEs load; Qwen is default
1K and 2K aligned presets generate
seed/steps/CFG/sampler/scheduler/denoise are runtime-patchable
output uploads to R2
pod remains reusable
```

Then add D1-ID LoRA disk/hot-RAM cache and style-reference mode before frontend rollout.

# SceneBuilder Krea 2 Runtime Plan

Status: **architecture / implementation plan**

Scope: add Krea 2 Turbo image generation to SceneBuilder Storyboard and CharacterScreen using the existing GPU-runtime + SceneBuilder2 control-plane patterns already proven by H3 and Enhancer.

This file is the source of truth for the Krea 2 implementation. Do not start broad SceneBuilder refactors from this work; implement the runtime and provider path first, then connect the existing image-generation job/UI flow.

---

## 1. Locked product decisions

### Model/runtime

- Primary model: **Krea 2 Turbo INT8 ConvRot**.
- Runtime: **ComfyUI native Krea 2 support**.
- Text encoder: **Qwen3-VL-4B BF16** for the first production path.
  - ComfyUI's native Krea 2 text encoder is Qwen3-VL-4B (`CLIPLoader` type `krea2`).
  - BF16 is intentionally preferred for our first build because CharacterScreen/Storyboard may use visual references and BF16 is the safest full-precision encoder path.
  - The encoder must be offloaded/unloaded after conditioning on lower-VRAM workers before diffusion starts.
  - FP8 text encoder may be added later as an optional memory/speed profile after canary testing; it is not required for v1.
- VAE is user-selectable:
  - `qwen_image` -> `qwen_image_vae.safetensors` (**default / official Comfy Krea path**)
  - `wan_2_1` -> selected Wan 2.1 VAE checkpoint (**alternate / canary before default exposure**)
- Minimum GPU: **20 GB VRAM**.
- Providers: **Novita GPU instances + RunPod Pods**, using the same provider-account variables and lifecycle style as H3/Enhancer.
- One generation at a time per GPU for v1.
- Output is uploaded to the existing project R2 image storage and finalized through the existing `image_generation_jobs` flow.

### LoRAs

- **0 to 3 LoRAs per generation.**
- LoRAs are **never baked into the Docker image**.
- Existing Krea 2 LoRAs already uploaded to R2 are the initial catalog entries:
  - `models/lora/krea2/MinimalisticVectorArtKrea2.safetensors`
  - `models/lora/krea2/Darkchurch_style_krea2_v1.0.safetensors`
- Existing thumbnails are UI assets only:
  - `models/lora/krea2/minimalist_vector_art_thumbnail.png`
  - `models/lora/krea2/dark_church_style_thumbnail.png`
- Thumbnails are never downloaded by the GPU runtime and never participate in inference.
- Runtime LoRA weights must stay **in CPU RAM / host memory, not local persistent disk**.
- Do **not** create `/opt/.../runtime-loras/<sha>.safetensors` files.
- Do **not** require SHA-256 naming for the two existing R2 files.
- An optional nullable `sha256` metadata field can exist later for integrity/deduplication, but the source of truth is the D1 LoRA ID + exact R2 object key. Existing objects do not need to be renamed or re-uploaded.

### Auth/config reuse

Reuse the same variables/secrets already used by H3/Enhancer wherever applicable:

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

Continue deriving a per-worker `SCENEBUILDER_POD_TOKEN` from `H3_POD_AUTH_MASTER_SECRET`. Do not create a second Krea-only master secret.

---

## 2. What current Krea 2 / ComfyUI support tells us

Current ComfyUI has native Krea 2 support and a native Krea 2 text encoder implementation based on Qwen3-VL-4B. The official Comfy Krea examples expose Krea 2 Turbo as an 8-step distilled model and support Krea style LoRAs.

Official Krea Turbo reference settings:

```text
variant: Krea 2 Turbo
steps: 8
CFG in official code: 0.0
mu: 1.15
resolution: 1K to 2K class outputs
```

The native Comfy workflow expresses the no-guidance Turbo setup as:

```text
steps: 8
cfg: 1.0
sampler: euler
scheduler: simple
denoise: 1.0
negative: ConditioningZeroOut from positive conditioning
latent: EmptyLatentImage
```

For our v1 runtime, the official/native Comfy workflow is the source of truth. Do not add an extra sampling-shift node just because the standalone Krea inference CLI exposes `mu=1.15`; first reproduce the official Comfy template exactly and only patch sampling if a controlled comparison proves it is needed.

Community reports worth preserving as canary notes, not hard defaults:

- INT8 ConvRot is currently a strong Krea 2 quality/speed choice on modern NVIDIA GPUs and is available in the Comfy-Org Krea 2 model set.
- CUDA 13 / cu130 is already being used successfully with Krea 2 INT8 ConvRot in current ComfyUI environments.
- PyTorch `2.13.0+cu130` appears in current Comfy/Krea INT8 issue reports and works as a valid environment baseline.
- Some users prefer Raw + Turbo LoRA around strength `0.6` and around `12` steps, but that is a separate profile from direct Turbo and is not our initial production path.
- Qwen Image VAE is the official Comfy Krea path. Wan 2.1 VAE is a community-proven alternative and should remain selectable only after our own A/B validation.
- INT8 ConvRot behavior can vary by GPU architecture and Comfy backend, so a real provider/GPU canary matrix is mandatory before enabling every >=20 GB card.

---

## 3. CUDA / PyTorch / Python baseline

Start from the same proven foundation as H3 unless Krea canaries show a concrete incompatibility:

```text
CUDA image: nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04
CUDA/PyTorch index: cu130
PyTorch: 2.13.0
Python: 3.13 target
Python 3.12: fallback only if a required Comfy/Krea dependency reproducibly fails on 3.13
```

Reasoning:

- H3 already runs CUDA 13.0 + PyTorch 2.13 in this repository.
- Current Comfy Krea 2 INT8 ConvRot reports include PyTorch 2.13 + cu130.
- Native Krea 2 support means we should avoid unnecessary custom dependency stacks.

Krea must use a **tested pinned ComfyUI commit/version** that contains native Krea 2 + INT8 ConvRot support. Do not depend indefinitely on an unpinned moving `master` in the final production image.

Also pin the compatible `comfy-kitchen` version pulled by that tested Comfy commit because INT8 ConvRot execution depends on its quantized kernels/backends.

---

## 4. Docker layering: optimize for cheap workflow/node changes

The Krea Docker chain must be layered so that changing workflow/runtime code does **not** redownload/rebuild the model weights.

The custom-node layer belongs **after all heavy model/VAE layers and before workflow**, as requested.

Recommended order:

```text
00 base
   CUDA 13.0 + Python + PyTorch 2.13 + system packages

10 comfyui
   pinned ComfyUI + Python requirements

20 krea2-model
   Krea 2 Turbo INT8 ConvRot diffusion model

30 vaes
   Qwen Image VAE
   Wan 2.1 VAE selected for our production comparison

40 text-encoder
   Qwen3-VL-4B BF16

50 nodes
   only Krea-specific/runtime helper nodes actually required
   includes SceneBuilder in-memory R2 LoRA loader

60 runtime
   job contract, R2 helper, workflow patcher, output upload,
   progress/cancellation/runtime logic

70 workflow
   baked API-format Krea 2 workflow + manifest only

80 pod
   HTTP pod server / entrypoint / readiness / lifecycle
```

Why this order:

- Changing Krea workflow only rebuilds `workflow` + `pod`.
- Changing runtime code rebuilds `runtime` + `workflow` + `pod`.
- Changing a custom node rebuilds `nodes` and the small layers after it, **not model/VAE downloads**.
- Changing the text encoder does not force the main Krea model or VAE layers to be rebuilt.
- LoRA additions never rebuild anything because LoRAs live in R2.

Potential target names:

```text
krea2-base
krea2-comfyui
krea2-model-int8
krea2-vaes
krea2-qwen-bf16
krea2-nodes
krea2-runtime
krea2-workflow
krea2-pod
```

Final image:

```text
khuxaima/scenebuilder-krea2-pod:latest
```

---

## 5. Hetzner layer-by-layer build

Add a dedicated workflow rather than overloading H3 or Enhancer build targets:

```text
.github/workflows/hetzner-krea2-build.yml
krea2/scripts/remote_build.sh
```

Mirror the mature H3/Enhancer temporary-builder behavior:

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

Build one dependent batch on one temporary Hetzner server so BuildKit cache survives between targets during that workflow run. Push every successful layer/target to Docker Hub before moving on.

Suggested first heavy build batch:

```text
base,comfyui,krea2-model-int8,vaes,qwen-bf16,nodes,runtime,workflow,pod
```

After those parents exist, normal workflow/runtime edits should rebuild only the final lightweight targets.

---

## 6. Comfy workflow contract

Files:

```text
krea2/workflows/krea2_turbo.json
krea2/workflows/manifests/krea2_turbo.json
```

Use a baked API-format workflow plus a manifest of patchable paths, following the H3 pattern. Do not search nodes dynamically by title at runtime.

Baseline graph should be as close as possible to the official native Comfy Krea 2 Turbo INT8 graph:

```text
UNETLoader
  krea2_turbo_int8_convrot.safetensors

CLIPLoader
  qwen3vl_4b_bf16.safetensors
  type = krea2

optional in-memory LoRA patch 1
optional in-memory LoRA patch 2
optional in-memory LoRA patch 3

CLIPTextEncode
ConditioningZeroOut
EmptyLatentImage
KSampler
VAEDecode
SaveImage / runtime output node
```

VAE path is patched per request:

```text
qwen_image -> qwen_image_vae.safetensors
wan_2_1   -> selected Wan 2.1 VAE filename
```

The manifest must expose at minimum:

```text
prompt
width
height
seed
steps
cfg
sampler
scheduler
denoise
diffusionModel
textEncoder
vae
outputPrefix
```

LoRA application does not need to be represented as a local-file `LoraLoader` node. The SceneBuilder in-memory LoRA helper can patch the loaded MODEL/CLIP objects before the sampler/prompt nodes execute.

---

## 7. Supported settings / API contract

Initial provider-neutral payload:

```json
{
  "jobId": "job_123",
  "projectId": "proj_123",
  "taskFamily": "krea2_image",
  "model": "krea-2-turbo",
  "prompt": "...",
  "width": 1536,
  "height": 1024,
  "settings": {
    "seed": 12345,
    "steps": 8,
    "cfg": 1.0,
    "sampler": "euler",
    "scheduler": "simple",
    "denoise": 1.0,
    "textEncoder": "qwen3vl_4b_bf16",
    "vae": "qwen_image",
    "loras": [
      { "id": "minimalistic-vector-art-krea2", "weight": 0.8 }
    ]
  },
  "inputs": {
    "outputPrefix": "projects/proj_123/images/generated"
  }
}
```

### v1 defaults

```text
steps: 8
cfg: 1.0
sampler: euler
scheduler: simple
denoise: 1.0
seed: random unless user pins it
text encoder: qwen3vl_4b_bf16
VAE: qwen_image
LoRAs: none
```

### v1 validation

- `loras.length <= 3`
- each LoRA ID must exist in D1, be enabled and support the selected model
- each LoRA weight must be inside that catalog entry's allowed strength range
- dimensions must be multiples of 8
- Krea 2 production size policy should stay in the **1K-2K class** initially
- reject absurd width/height at the Worker before provisioning GPU capacity
- steps are constrained by model/LoRA policy rather than arbitrary browser values
- Turbo default remains 8 steps; higher values are advanced only

### Advanced settings policy

Expose only settings we actually validate on our Docker image. ComfyUI may technically offer many samplers/schedulers, but SceneBuilder should not advertise untested combinations.

Initial advanced candidates to benchmark:

```text
steps: 4-20 (8 default)
sampler: euler first
scheduler: simple first
VAE: qwen_image | wan_2_1
seed: integer/random
LoRA strengths: per-catalog range
```

Do not expose every Comfy sampler just because it exists. Add alternatives after matched-seed canaries.

---

## 8. LoRA architecture: R2 -> CPU RAM -> Comfy model patch

This is a hard requirement: **no LoRA local-disk cache**.

### Browser request

Browser sends only:

```json
[
  { "id": "minimalistic-vector-art-krea2", "weight": 0.8 },
  { "id": "dark-church-krea2", "weight": 0.65 }
]
```

The browser never sends:

- an arbitrary R2 key
- a filesystem path
- a random URL
- R2 credentials

### Worker resolution

For every requested LoRA:

```text
LoRA id
  -> D1 lora row
  -> D1 model-support row
  -> validate enabled/type/model/quantization/steps/strength
  -> trusted exact R2 object key
  -> send trusted LoRA descriptor to pod
```

Pod payload may contain the resolved object key because the pod already has private R2 runtime credentials.

### Pod loading

Implement a SceneBuilder Krea LoRA helper/custom node that avoids `folder_paths` and local files:

```text
R2 GetObject
  -> read/stream object bytes into host CPU memory
  -> safetensors.torch.load(bytes)
  -> CPU tensor/state dict
  -> comfy.sd.load_lora_for_models(...)
  -> patched MODEL (and CLIP only if catalog says so)
```

Current Comfy `load_lora_for_models()` accepts a LoRA state dictionary, so the runtime does not need the standard filename-based `LoraLoader` path.

For Krea 2 style LoRAs, default target is:

```text
model strength = requested weight
clip strength = 0
```

because the official native Krea workflow uses `LoraLoaderModelOnly` for its style LoRAs.

Chain at most three patches in deterministic request order.

After patches are materialized safely:

- release the raw downloaded byte buffer
- release temporary unneeded state dictionaries when Comfy no longer needs them
- keep remaining patch tensors in CPU RAM/offload memory as required by Comfy
- never persist the safetensors payload to pod disk

If future Comfy internals require a file path, the only acceptable fallback is a RAM-backed `tmpfs`/`/dev/shm` path, never the container's persistent filesystem. The preferred implementation remains direct `safetensors.torch.load(bytes)`.

### Warm-worker behavior

Do not build a disk LoRA cache. A warm worker may retain already-loaded LoRA CPU state in a bounded in-process RAM cache keyed by D1 LoRA ID/object key if benchmarks show it is useful.

Requirements:

- strict RAM cap / LRU eviction
- cache is process-local and disposable
- no correctness dependency on cache
- clear cache on memory pressure
- job cancellation/failure must not leave unbounded RAM retained

---

## 9. Generic D1 LoRA catalog

Create a table named exactly:

```text
lora
```

It represents one physical LoRA asset. One physical `.safetensors` file can support many models without duplicating the R2 object.

Suggested schema:

```sql
CREATE TABLE IF NOT EXISTS lora (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
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

    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
```

`lora_type` values should support future use:

```text
image
video
both
```

`thumbnail_object_key` is for UI only.

`sha256` is nullable optional metadata. It is **not** the object name and is **not required** to use the two existing Krea files.

### Model compatibility table

Do not store `model_supported` as one comma-separated field. Use a many-to-many compatibility table so one LoRA can support Krea Turbo, Krea Raw, Z-Image Turbo, future image models, or future video models.

```sql
CREATE TABLE IF NOT EXISTS lora_model_support (
    lora_id TEXT NOT NULL,
    model_key TEXT NOT NULL,

    quantizations_json TEXT NOT NULL DEFAULT '[]',

    min_strength REAL,
    max_strength REAL,
    default_strength REAL,

    min_steps INTEGER,
    max_steps INTEGER,
    recommended_steps INTEGER,

    apply_target TEXT NOT NULL DEFAULT 'model_only',
    notes TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,

    PRIMARY KEY (lora_id, model_key),
    FOREIGN KEY (lora_id) REFERENCES lora(id)
);
```

Example `quantizations_json`:

```json
["int8_convrot", "bf16", "fp8_scaled"]
```

`apply_target` supports future LoRA kinds:

```text
model_only
clip_only
model_and_clip
```

This is flexible enough for future image/video LoRAs without moving or duplicating their physical R2 files.

### Initial Krea 2 catalog rows

Seed D1 against the already-uploaded objects; do not upload them again:

```text
minimalistic-vector-art-krea2
  object: models/lora/krea2/MinimalisticVectorArtKrea2.safetensors
  thumbnail: models/lora/krea2/minimalist_vector_art_thumbnail.png
  type: image
  model support: krea-2-turbo

dark-church-krea2
  object: models/lora/krea2/Darkchurch_style_krea2_v1.0.safetensors
  thumbnail: models/lora/krea2/dark_church_style_thumbnail.png
  type: image
  model support: krea-2-turbo
```

Exact recommended strength/min/max/step ranges should be filled from each LoRA's known training/release settings or our A/B canary; do not invent them in the migration.

---

## 10. GPU policy

Use the existing Enhancer GPU catalog as the base inventory instead of maintaining another unrelated GPU-name list.

Krea rule:

```text
KREA2_MIN_VRAM_GB = 20
```

Initial RunPod candidates from the current catalog include >=20 GB cards such as:

```text
RTX 4000 Ada 20 GB
RTX A4500 20 GB
L4 24 GB
RTX 4090 24 GB
RTX A5000 24 GB
RTX 3090 24 GB
RTX PRO 4000 Blackwell 24 GB
RTX 5090 32 GB
RTX PRO 4500 / 4500 SE 32 GB
L40 48 GB
L40S 48 GB
RTX 6000 Ada 48 GB
A40 48 GB
RTX A6000 48 GB
```

Novita current candidates are >=24 GB in the existing catalog:

```text
RTX 4090
RTX 5090
RTX 6000 Ada
L40S
```

Use the same nominal-VRAM tolerance as Enhancer because provider telemetry may report a nominal 20 GB card a few MiB below the exact binary boundary.

### Architecture caveat

INT8 ConvRot performance is not equally good on every NVIDIA architecture. The router may start from the Enhancer list, but production eligibility requires a Krea canary pass per GPU class.

Maintain a Krea qualification result such as:

```text
supported
supported_but_slow
unsupported
```

Do not assume an A100/datacenter Ampere result is representative of 4090/Ada or 5090/Blackwell INT8 performance.

### OOM escalation

If a valid job OOMs after the required encoder unload/offload path:

```text
20 GB -> 24 GB -> 32 GB -> 48 GB
```

Record the escalation in job routing metadata and do not repeatedly retry the same VRAM tier.

---

## 11. Host RAM / offload policy

Because v1 intentionally uses a BF16 Qwen3-VL-4B encoder and RAM-only LoRAs, host RAM is part of worker qualification.

Initial target:

```text
minimum host RAM: 32 GB
prefer >=48 GB for 20/24 GB GPU workers when provider inventory allows
```

Runtime sequence on constrained GPUs:

```text
1. load/activate Qwen text encoder
2. encode prompt / references
3. offload or unload Qwen encoder from VRAM
4. free CUDA cache
5. load/apply Krea model patches / LoRAs
6. sample Krea Turbo
7. decode selected VAE
8. upload result
```

Do not let host-memory LoRA loading silently spill to a persistent local file cache.

---

## 12. RunPod / Novita pod lifecycle

Follow H3/Enhancer GPU-instance provisioning, not the older endpoint-per-GPU serverless registry pattern.

Pod API should reuse the proven shape:

```text
GET  /health
GET  /ready

POST /workloads
GET  /workloads/:id
POST /workloads/:id/cancel
POST /workloads/:id/items/:jobId/cancel
```

Protected endpoints use:

```text
Authorization: Bearer <SCENEBUILDER_POD_TOKEN>
```

Events back to SceneBuilder:

```text
worker_ready
workload_started
job_progress
job_completed
job_failed
job_cancelled
worker_idle
idle_expired
```

Recommended Krea progress phases:

```text
queued
allocating
cold_start
preparing_model
loading_loras
encoding_prompt
generating
decoding
uploading
completed
failed
cancelled
```

Cancellation should call Comfy `/interrupt` for an active generation and then clean job-specific RAM objects.

Idle workers should be reusable for additional Krea jobs until the configured idle deadline.

Start with one active generation per GPU. Do not add same-GPU concurrent Krea requests until memory/performance benchmarks justify it.

---

## 13. SceneBuilder2 integration

Do not build a separate React networking path for Krea.

Reuse the existing central image-generation path:

```text
CharacterScreen / Storyboard
  -> apiGenerateImage()
  -> /api/generate-image
  -> durable image_generation_jobs row
  -> async status polling
  -> existing cancel route
  -> normal R2 image finalization
  -> existing thumbnail flow
```

Extend model/provider handling with a model key such as:

```text
krea-2-turbo
```

The browser never talks directly to RunPod/Novita.

### Existing `image_generation_jobs`

Reuse it as the durable per-image job table. Add only generic fields needed for GPU routing if not already available, for example:

```text
requested_provider
actual_provider
pod_worker_id
settings_json
provider_detail_json
attempt_log_json
started_at
heartbeat_at
```

`settings_json` should preserve the immutable generation snapshot:

```text
model key
quantization
width/height
seed
steps
sampler/scheduler
VAE
text encoder profile
LoRA ids + strengths
reference-mode settings when present
requested compute backend
```

### UI

Create one shared LoRA selector used by both Storyboard and CharacterScreen:

```text
<LoraSelector modelKey="krea-2-turbo" maxSelected={3} />
```

Catalog endpoint concept:

```text
GET /api/loras?model=krea-2-turbo&type=image
```

Return UI data only:

```text
id
name
description
thumbnail URL
trigger words
min/max/default strength
recommended steps
```

Do not return private R2 credentials or runtime secrets.

---

## 14. Reference-image mode

Krea 2's Qwen3-VL text encoder is vision-capable and current ComfyUI also has a Krea 2 style-reference workflow using the INT8 model plus a dedicated style-reference LoRA.

For SceneBuilder:

- Text-to-image Krea v1 must work first without references.
- Then add CharacterScreen/Storyboard visual-reference support as a second canary.
- Reference mode must still obey the **no baked LoRA** rule.
- If the dedicated Krea reference LoRA is required, store it in R2 and register it as a hidden/internal `lora` row rather than baking it into Docker.
- Keep BF16 Qwen3-VL-4B as the initial encoder for this path unless our tested Comfy build proves its lower-precision vision path is fully stable.
- Start with the official Comfy reference behavior/limits before generalizing arbitrary multi-image conditioning.

---

## 15. Output contract

The GPU runtime should return a normal SceneBuilder image result, for example:

```json
{
  "ok": true,
  "jobId": "job_123",
  "projectId": "proj_123",
  "runtime": "krea2-pod",
  "outputs": {
    "image": {
      "objectKey": "projects/proj_123/images/generated/job_123.png",
      "url": "https://...",
      "uploaded": true
    }
  },
  "settings": {
    "seed": 12345,
    "vae": "qwen_image"
  }
}
```

The SceneBuilder Worker remains responsible for durable job state/ownership and normal UI thumbnail bookkeeping.

---

## 16. Runtime repository layout

Target layout:

```text
krea2/
  plan.md
  README.md

  docker/
    Dockerfile.base
    Dockerfile.comfyui
    Dockerfile.krea2-model-int8
    Dockerfile.vaes
    Dockerfile.qwen-bf16
    Dockerfile.nodes
    Dockerfile.runtime
    Dockerfile.workflow
    Dockerfile.pod

  workflows/
    krea2_turbo.json
    manifests/
      krea2_turbo.json

  src/
    common/
      job_contract.py
      runtime.py
      r2.py
      lora.py
      gpu.py
    custom_nodes/
      scenebuilder_krea2/
        __init__.py
        r2_lora.py
    pod/
      server.py

  scripts/
    remote_build.sh
    smoke_test.py
```

Do not copy unrelated H3 model logic into Krea. Reuse patterns/contracts, then extract genuinely generic helpers only where it reduces duplication without risking H3.

---

## 17. Tests / canary matrix

### Build smoke

```text
CUDA runtime visible
PyTorch == 2.13.x + cu130
ComfyUI starts
comfy-kitchen INT8 backend available
Krea 2 native model class recognized
Qwen3-VL-4B `krea2` CLIP type recognized
both VAEs load
```

### Model canary

Matched prompt/seed/dimensions:

```text
Krea INT8 ConvRot
8 steps
CFG 1
Euler / simple
Qwen Image VAE
no LoRA
```

Then repeat with Wan 2.1 VAE and compare:

```text
peak VRAM
host RAM
load time
sampling time
decode time
output quality/color/detail
```

### LoRA canary

Test the two existing R2 LoRAs without touching Docker:

```text
0 LoRA
MinimalisticVectorArt only
Darkchurch only
both together
3-LoRA synthetic/catalog test once a third compatible LoRA exists
```

Verify:

```text
R2 -> RAM only
no safetensors appears on local persistent filesystem
weights unload/evict correctly
cancellation frees RAM
warm second job does not accumulate unbounded memory
```

### GPU/provider matrix

At minimum:

```text
RunPod 20 GB eligible card
RunPod RTX 4090 / 24 GB
RunPod RTX 5090 / 32 GB
Novita RTX 4090
Novita RTX 5090
```

Also benchmark any 48 GB classes likely to be selected by fallback.

For each class record:

```text
cold start
model load
peak VRAM
peak host RAM
8-step generation time
2K generation behavior
LoRA behavior
OOM/retry result
```

### Lifecycle tests

```text
job cancel while queued
job cancel during sampling
pod dies during generation
callback loss / retry
R2 read failure
R2 upload failure
invalid LoRA ID
LoRA incompatible with model
LoRA weight outside allowed range
4 LoRAs rejected
20 GB OOM escalates exactly once per tier
idle worker reuse
idle worker deletion
```

### SceneBuilder tests

```text
CharacterScreen single generation
CharacterScreen LoRA generation
Storyboard single scene
Storyboard multi-scene queue
status polling
cancel
reload while generation is running
completed R2 image writeback
thumbnail creation
saved LoRA/settings snapshot remains reproducible
```

---

## 18. Rollout order

### Phase 1 - runtime proof only

1. Build Krea base with CUDA 13 / PyTorch 2.13.
2. Pin tested ComfyUI.
3. Bake Krea 2 Turbo INT8 ConvRot.
4. Bake both VAEs.
5. Bake Qwen3-VL-4B BF16.
6. Reproduce official native Comfy Turbo output at 8 steps / CFG 1 / Euler simple.
7. Pass 20 GB GPU canary with encoder offload.

Do not touch frontend routing until this works.

### Phase 2 - RAM-only R2 LoRAs

1. Implement direct R2 -> bytes -> `safetensors.torch.load(bytes)` loader.
2. Patch model through Comfy `load_lora_for_models()`.
3. Apply 0-3 LoRAs in order.
4. Test both existing R2 LoRAs.
5. Prove no local persistent `.safetensors` file is created.

### Phase 3 - provider pods

1. Add Krea pod HTTP runtime.
2. Reuse H3 auth derivation and R2/provider env.
3. Add RunPod allocator using Enhancer >=20 GB catalog.
4. Add Novita allocator.
5. Add worker readiness/qualification and OOM escalation.
6. Add warm reuse and idle delete.

### Phase 4 - D1 catalog/control plane

1. Add `lora` table.
2. Add `lora_model_support`.
3. Seed the two already-uploaded Krea LoRAs.
4. Add catalog API filtered by model/type.
5. Extend durable image jobs with Krea routing/settings metadata.

### Phase 5 - SceneBuilder UI

1. Add `krea-2-turbo` to existing image model selection.
2. Add shared max-3 LoRA selector to CharacterScreen + Storyboard.
3. Add Qwen/Wan VAE selection where appropriate.
4. Keep current async polling/cancel/thumbnail behavior.

### Phase 6 - reference images

1. Add the official Krea reference workflow as a separate manifest/profile.
2. Put any required reference LoRA in R2, not Docker.
3. Validate BF16 Qwen vision path on 20/24/32 GB workers.
4. Enable CharacterScreen/Storyboard reference inputs only after that canary passes.

---

## 19. Implementation guardrails

- Never bake user/style LoRAs into a Krea Docker image.
- Never use local persistent disk as the normal LoRA cache.
- Never allow the browser to choose an arbitrary R2 object key/path.
- Never accept more than 3 LoRAs.
- Never trust browser strength/step values without D1 compatibility validation.
- Never expose R2 credentials to the browser.
- Never create a new image-generation UI polling system when the existing one works.
- Never fork H3 auth secrets just for Krea.
- Never add every Comfy sampler/scheduler to the UI without our own canary.
- Never assume all >=20 GB GPU families perform INT8 ConvRot equally well.
- Keep heavy weights below the `nodes`, `runtime` and `workflow` layers so ordinary code/workflow edits stay cheap.

---

## 20. First implementation milestone

The first milestone is intentionally narrow:

```text
Krea 2 Turbo INT8 ConvRot
+ Qwen3-VL-4B BF16
+ Qwen Image VAE
+ CUDA 13 / PyTorch 2.13
+ official-like 8-step native Comfy workflow
+ one 20 GB RunPod canary
+ no LoRA yet
```

Acceptance criteria:

```text
container builds through Hetzner layer-by-layer
Comfy boots cleanly
model/encoder/VAE load
1024-class generation succeeds
2K-class generation succeeds or produces a documented VRAM escalation
encoder leaves GPU before diffusion on constrained worker
peak VRAM is recorded
output uploads to R2
pod remains reusable for a second generation
```

Then add RAM-only R2 LoRAs as Phase 2, before SceneBuilder UI work.

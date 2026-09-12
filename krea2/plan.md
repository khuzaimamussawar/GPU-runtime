# SceneBuilder Krea 2 Runtime Plan

Status: **architecture / implementation plan**

Scope: add Krea 2 Turbo image generation to SceneBuilder Storyboard and CharacterScreen while creating a **generic image-generation pod/control-plane layer** that can host additional image models later. Krea 2 is the first image family; it must not become the name or boundary of the reusable pod infrastructure.

This file is the source of truth for the Krea 2 runtime, generic image-pod lifecycle, D1 scheduling, LoRA catalog, and Storyboard/Character integration.

---

## 1. Locked product decisions

### Krea 2 model/runtime

- Primary model: **Krea 2 Turbo INT8 ConvRot**.
- Diffusion checkpoint: `krea2_turbo_int8_convrot.safetensors`.
- Runtime: **native ComfyUI Krea 2 support**.
- Text encoder: **Qwen3-VL-4B BF16** (`qwen3vl_4b_bf16.safetensors`).
- Qwen3-VL-4B handles normal text conditioning and native image/style-reference conditioning.
- Default production resolution: **2048x1152 landscape / 1152x2048 portrait**.
- Optional fast/1K preset: **1280x720 landscape / 720x1280 portrait**.
- Minimum GPU: **20 GB VRAM**.
- Providers: **RunPod Pods + Novita GPU instances**.
- One pod runs **one image generation at a time**. No concurrent images inside one GPU pod.
- A warm pod is reusable immediately after an image finishes.
- Pods are a **global compatible pool**, not owned by one project.
- Maximum **5 simultaneously active/reserved image pods per project**.
- New capacity target: **1 pod per 3 ready queued images**, capped at 5 pods per project.
- Reuse the same RunPod/Novita/R2 variables and H3 pod-auth master secret already used by the existing GPU control plane.

### Storage

- **No network volume.**
- **No network disk.**
- **No provider-mounted model storage.**
- Provider root/container disk: **35 GB**.
- Krea model, Qwen encoder, VAEs, and the native Krea style-reference adapter are baked into Docker image layers.
- User-selectable LoRAs are **not** baked into Docker.
- User LoRAs may be downloaded from R2 and cached on the local 35 GB container disk.
- Generated inputs/outputs may use temporary local files during a job, then are cleaned after upload/finalization.

Provider configuration:

```text
IMAGE_POD_DISK_GB = 35
RunPod: containerDiskInGb = 35
Novita: rootfsSize = 35
Novita: networkStorages = []
RunPod: no network volume / volume mount
```

The final image must leave writable headroom for:

```text
Comfy input/output/temp
atomic user-LoRA downloads
bounded user-LoRA cache
logs / runtime scratch
```

Use measured free-space watermarks instead of assuming a fixed cache capacity before the final image is measured.

---

## 2. Native Krea decisions from official + community evidence

### Resolution: lock 2048x1152 / 1152x2048 as the normal default

Do **not** use `1368x768` as the product default.

Official Krea 2 Turbo is designed for roughly **1K through 2K** generation, and the official sampler pads width/height upward to its required alignment when a dimension is not aligned. The distilled Turbo checkpoint is explicitly intended for high-quality few-step inference up to the 2K range.

For SceneBuilder 16:9 / 9:16, use:

```text
NORMAL / QUALITY
landscape: 2048 x 1152
portrait:  1152 x 2048

FAST / 1K OPTIONAL
landscape: 1280 x 720
portrait:  720 x 1280
```

Why the normal default is `2048x1152` rather than `1280x720` or `1368x768`:

- `2048x1152` is exact 16:9 and both dimensions are divisible by 16.
- `1152x2048` is the exact portrait equivalent.
- the long edge is exactly the official 2K ceiling;
- it avoids hidden alignment padding;
- recent community Krea 2 Turbo usage repeatedly uses `2048x1152` / `1152x2048` successfully;
- community reports also tend to show stronger detail/diversity at higher native Krea 2 Turbo resolutions;
- `1368x768` is only a near-1MP calculator shape and `1368` is not divisible by 16, so the runtime would pad it anyway;
- `1280x720` works technically and is useful as a faster/lower-memory option, but it is not the quality default.

There is **no resolution A/B benchmark requirement** in the rollout plan. The production decision is already made:

```text
DEFAULT = 2048x1152 / 1152x2048
FAST    = 1280x720 / 720x1280
```

The runtime still performs a functional smoke test at both tiers to prove memory/runtime correctness, not to choose between them.

### Turbo sampler baseline

Native/production baseline:

```text
steps:     8
cfg:       1.0 in Comfy UI contract / effectively no-guidance Turbo path
sampler:   euler
scheduler: simple
denoise:   1.0
mu:        native Turbo behavior / 1.15 where the workflow exposes it
seed:      random unless user pins one
```

These are defaults, not hard-coded limits. User advanced controls remain available.

### INT8 ConvRot choice

Keep **INT8 ConvRot** as the v1 checkpoint. Recent community comparisons put it among the strongest quality/speed quantized Krea 2 Turbo options and it fits the direction already chosen for SceneBuilder. Do not introduce INT4/GGUF as a second v1 variable.

---

## 3. VAE

User-selectable:

```text
qwen_image -> qwen_image_vae.safetensors
wan_2_1    -> selected Wan 2.1 VAE checkpoint
```

Rules:

- `qwen_image` is the default/native Krea 2 path.
- `wan_2_1` is the alternate selectable VAE.
- Qwen text conditioning is independent of the selected VAE decode path.
- Both VAE files are baked into the image.
- Runtime validation only needs to prove both decode correctly on the final pinned stack; no quality tournament is required.

---

## 4. User LoRAs and baked style-reference adapter

### User LoRAs

- Normal LoRA mode supports **0-3 user-selected LoRAs**.
- Existing R2 objects remain unchanged:

```text
models/lora/krea2/MinimalisticVectorArtKrea2.safetensors
models/lora/krea2/Darkchurch_style_krea2_v1.0.safetensors
```

- Existing thumbnails are UI-only assets:

```text
models/lora/krea2/minimalist_vector_art_thumbnail.png
models/lora/krea2/dark_church_style_thumbnail.png
```

- Every user LoRA gets an immutable D1 `lora_id`.
- Browser/API/job identity is `loraId`, never display name or filename.
- Display names are intentionally not unique.
- D1 resolves `lora_id -> trusted exact R2 object key` before dispatch.
- Optional SHA-256/ETag metadata is integrity/version metadata, not identity.

### Native style-reference adapter is baked

The native Comfy Krea 2 style-reference workflow requires:

```text
krea2_style_reference.safetensors
```

Bake this system adapter into the Docker image. It is approximately 457 MB and is a fixed runtime dependency, so there is no benefit in fetching it from R2 for every fresh pod.

Bake it at:

```text
/opt/scenebuilder-models/krea2/loras/krea2_style_reference.safetensors
```

Comfy sees it through the configured extra model paths.

Rules:

- it is a **system/internal runtime asset**, not a user catalog LoRA;
- it is not returned by `GET /api/loras`;
- it does not consume one of the user's 0-3 LoRA slots;
- style-reference mode uses it automatically;
- do not create an R2 runtime download/cache path for it;
- no D1 `lora` row is required for the baked system adapter; its version/checksum belongs in runtime build metadata if we want observability.

---

## 5. Native ComfyUI baseline

Use the same tested Comfy revision currently used by H3 unless a real Krea runtime incompatibility requires a deliberate shared-pin upgrade:

```text
COMFYUI_COMMIT = 2a68ce33b4c9ea6ee4283e618a74560cefb32694
H3 label: v0.31.0-9-g2a68ce33
comfy-kitchen = 0.2.28
```

That revision already contains:

- native Krea 2 model detection/model class;
- native `CLIPType.KREA2`;
- Krea-specific Qwen3-VL-4B conditioning;
- Qwen3-VL image preprocessing;
- INT8/ConvRot quantization support through `comfy-kitchen`;
- CUDA 13 optimized quantized backend support.

Do not follow moving Comfy `master` in production.

---

## 6. CUDA / PyTorch / Python baseline

Start with the same proven software family as H3:

```text
CUDA: 13.0 / cu130
PyTorch: 2.13.0
Python: 3.13 target
Ubuntu: 24.04
```

The pinned Comfy quantization path supports the optimized `comfy-kitchen` CUDA operations required by the selected INT8 ConvRot route.

Build stages may use CUDA devel when required. Prefer a matching CUDA 13 runtime image for the final stage if the real ConvRot smoke test passes.

Strip build-only artifacts from final layers:

```text
apt lists
pip cache
HF cache
git history
compiler/build scratch
```

No SageAttention or FlashAttention dependency in v1. Start with native Comfy/PyTorch attention plus `comfy-kitchen` ConvRot.

---

## 7. Docker layer order and why Comfy can come after the heavy weights

It is safe for the **Docker build lineage** to add Krea/Qwen/VAE files before installing Comfy. Docker layer order is not runtime execution order: the final container filesystem contains every lower layer before the runtime starts.

The unsafe pattern would be putting model files inside `/opt/ComfyUI` before cloning/installing Comfy into that same non-empty directory.

Use a Comfy-independent model root:

```text
/opt/scenebuilder-models/krea2/
  diffusion_models/
    krea2_turbo_int8_convrot.safetensors
  text_encoders/
    qwen3vl_4b_bf16.safetensors
  vae/
    qwen_image_vae.safetensors
    <wan-2.1-vae>.safetensors
  loras/
    krea2_style_reference.safetensors
```

Then install the single Comfy copy into:

```text
/opt/ComfyUI
```

and wire `/opt/scenebuilder-models/krea2` through Comfy `extra_model_paths.yaml` or equivalent stable folder-path registration.

We do **not** copy H3's historical two-Comfy/core-overlay retrofit.

Recommended linear chain:

```text
00 image-krea2-base
   CUDA 13 + Python + PyTorch 2.13/cu130 + common runtime packages

10 image-krea2-model-int8
   Krea 2 Turbo INT8

20 image-krea2-vaes
   Qwen Image + Wan 2.1 VAE

30 image-krea2-qwen-bf16
   Qwen3-VL-4B BF16

35 image-krea2-system-adapters
   baked krea2_style_reference.safetensors

40 image-krea2-comfyui
   one pinned Comfy revision
   matching requirements/comfy-kitchen
   extra model paths -> /opt/scenebuilder-models/krea2
   verify Krea2 + KREA2 CLIP + ConvRot + system LoRA discovery

50 image-krea2-nodes
   only helper/custom nodes actually required

60 image-krea2-workflow
   API workflows + manifests

70 image-krea2-runtime
   LAST layer
   generic image pod HTTP server
   model-family adapter registry
   R2/output handling
   user-LoRA cache manager
   workflow patching
   cancellation/progress/readiness
   memory cleanup/offload
   global-pool idle/draining lifecycle
```

Consequences:

- changing Comfy does not redownload parent Krea/Qwen/VAE/system-adapter layers;
- changing nodes rebuilds nodes/workflow/runtime only;
- changing workflows rebuilds workflow/runtime only;
- changing runtime rebuilds only the final layer;
- adding/changing user LoRAs changes D1/R2 only.

Final image may still be model-specific:

```text
khuxaima/scenebuilder-krea2-pod:latest
```

while the control-plane tables/protocol stay generic.

---

## 8. Hetzner layer-by-layer build

Add:

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

Targets:

```text
base
model-int8
vaes
qwen-bf16
system-adapters
comfyui
nodes
workflow
runtime
```

Push each successful parent so later retries reuse the published heavyweight layers.

---

## 9. Sampling/settings contract

User controls:

```text
prompt
prompt enhancement on/off
negative prompt
resolution tier / orientation
seed / random seed
steps
CFG
sampler
scheduler
denoise
VAE
0-3 user LoRAs OR style-reference images
strength for each selected user LoRA
```

Initial advanced ranges:

```text
steps:       1-50, default 8
cfg:         configurable, default 1.0 in Comfy contract
sampler:     allowlisted; Euler guaranteed
scheduler:   allowlisted; simple guaranteed
denoise:     0.0-1.0, default 1.0
seed:        explicit integer or random
VAE:         qwen_image | wan_2_1
resolution:  quality 2048x1152 or fast 1280x720, orientation mirrored
```

Do not silently accept arbitrary sampler/scheduler strings. Persist exact selected values in the durable job snapshot.

### Negative prompt

Native Turbo/no-guidance behavior effectively makes a typed negative prompt inactive at the default no-guidance/CFG1 Comfy path.

UI/runtime behavior:

- default native path: show negative prompt as inactive/no effect;
- if the user intentionally enables guided/higher-CFG mode, encode supplied negative text with the same Krea text encoder;
- persist both the chosen guidance mode and negative text.

### Prompt enhancement

Always store the raw user prompt. If enhancement is enabled, also store the effective/enhanced prompt in execution metadata.

---

## 10. Two style paths

Per generation choose one of:

```text
A. USER LoRA mode
   0-3 user-selected D1 LoRA IDs
   no style-reference images

B. STYLE REFERENCE mode
   zero user-selected LoRAs
   1-3 style-reference images
   baked krea2_style_reference.safetensors is applied automatically
```

Do not combine arbitrary user LoRAs and style-reference images in v1.

---

## 11. Style-reference image dimensions

Style references do **not** need to match:

```text
each other
output width/height
output aspect ratio
```

Do not stretch all references to the output shape.

Runtime preprocessing:

```text
1. EXIF-orient
2. convert to RGB
3. preserve source aspect ratio
4. do not force output aspect ratio
5. only downscale an input when it exceeds the runtime safety cap
6. let pinned native Qwen/Krea preprocessing align each image to its own patch grid
```

Functional tests must include square + landscape + portrait references and odd user dimensions. This is a compatibility smoke test, not a visual benchmark.

---

## 12. Generic D1 LoRA catalog

Create one catalog for future image and video LoRAs.

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

    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
```

Rules:

- `id` is immutable API/cache/job identity.
- `display_name` is **not unique**.
- `thumbnail_object_key` is UI-only.
- `lora_type`: `image | video | both`.
- this table is for catalog/dynamic assets; the baked Krea style-reference system adapter does not need a row.

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

One physical R2 asset can support multiple future model keys/quantizations without duplication.

---

## 13. User-LoRA local disk + hot RAM policy

Local cache path:

```text
/opt/scenebuilder-image/cache/loras/<lora_id>.safetensors
```

Use immutable `lora_id`, never display name, as cache filename/key.

First use:

```text
browser loraId
 -> Worker resolves D1 asset + compatibility
 -> pod receives trusted loraId + R2 key + integrity metadata
 -> cache hit: reuse local file
 -> cache miss: R2 -> <lora_id>.part -> validate -> atomic rename
 -> parse LoRA tensors
 -> apply model patch at requested strength
```

Hot CPU cache:

```text
identity = lora_id + asset version/etag/sha256 when known
```

Strength is not part of cache identity. The same parsed tensors can be repatched at another strength.

If the next job uses the same LoRA set, reuse hot CPU tensors. If it uses a different set, old tensors are evictable from RAM while the safetensors file remains in bounded local disk cache.

Disk eviction rules:

- never evict baked model/Qwen/VAE/system-adapter assets;
- never evict an active job's user LoRA;
- delete oldest unused cached user LoRAs first;
- redownload from R2 if needed later.

---

## 14. Krea/Qwen/VAE warm CPU offload policy

The main CPU-offload targets are the expensive **Krea model + Qwen text encoder**, not the LoRA file.

Desired post-job state:

```text
VRAM:
  job activations/workspaces/sampler allocations released
  reclaimable CUDA cache cleared

CPU RAM:
  Krea model object/weights retained where safe
  Qwen encoder object/weights retained where safe
  VAE object retained where useful
  current user-LoRA tensors retained if within hot-cache budget

Container disk:
  baked checkpoints + baked style adapter remain
  bounded dynamic user-LoRA cache remains
```

Constrained-GPU sequence:

```text
1. get Qwen from warm CPU state or baked checkpoint
2. move required encoder state to GPU
3. encode prompt + optional style images
4. offload Qwen back to CPU
5. free reclaimable GPU cache
6. get Krea from warm CPU state or baked checkpoint
7. activate Krea under Comfy memory management
8. get user LoRAs from hot RAM / disk / R2 when LoRA mode is used
9. apply user LoRAs OR baked style-reference adapter path
10. sample one image
11. activate selected VAE and decode
12. upload result
13. delete job temp media
14. retain reusable model objects in CPU RAM
15. retain useful dynamic LoRA tensors under RAM cap
16. release job-specific GPU memory
```

Do **not** run `unload_models=true` after every normal successful job. That destroys warm reuse.

Normal same-family cleanup frees reclaimable GPU memory while preserving reusable host/offloaded model state. Full model unload is appropriate for:

```text
incompatible model-family switch
fatal CUDA/model state
host-RAM pressure requiring eviction
pod draining/deletion
```

Host-RAM eviction order:

```text
old hot dynamic LoRA tensors
inactive VAE
Qwen warm state
Krea warm state last
```

---

## 15. Enhancer engine pattern to copy

Enhancer already demonstrates the lifecycle concept:

```text
persistent process cache
  reusable expensive engine/model state survives jobs

job execution cache
  contexts/streams/workspaces are releasable after each job
```

Generic image-pod analogue:

```text
warm/process state:
  model-family adapter
  Krea/Qwen/VAE CPU objects
  current dynamic LoRA parsed states
  workflow/model metadata

job-specific state:
  activations
  latents
  sampler tensors
  temporary patches
  reference tensors
  CUDA workspace/cache
```

Release the second category after every image; preserve the first as host RAM allows.

---

## 16. Generic image pod — not Krea-specific

Service identity:

```text
scene-builder-image-pod
```

Krea adapter/task family:

```text
krea2_image
```

Future examples:

```text
z_image_turbo
krea2_base
future_flux_family
other image families
```

A runtime image advertises supported families in `capabilities_json`. The control plane reuses a worker only when its image/capabilities support the requested family.

Runtime registry:

```text
TASK_FAMILY_REGISTRY = {
  "krea2_image": Krea2Adapter,
  ...future image adapters...
}
```

Each adapter owns:

```text
workflow + manifest selection
required files
request/settings validation
style/reference handling
model-switch cleanup rules
result parsing
minimum GPU/runtime requirements
```

The HTTP server, auth, one-job queue, idle timeout, draining, callbacks, diagnostics, provider lifecycle, and D1 worker table remain generic.

---

## 17. D1 `image_pod_workers` — global reusable worker inventory

Create a separate generic image-worker table. Do not reuse `h3_pod_workers` / `enhancer_pod_workers`, and do not create `krea2_pod_workers`.

```sql
CREATE TABLE IF NOT EXISTS image_pod_workers (
    id TEXT PRIMARY KEY,

    service_kind TEXT NOT NULL DEFAULT 'image_generation',

    -- current assignment metadata only; NOT permanent ownership
    project_id TEXT,
    user_email TEXT,

    provider TEXT NOT NULL,
    provider_instance_id TEXT,
    endpoint_url TEXT,
    region TEXT,

    gpu_class TEXT,
    provider_gpu_name TEXT,
    compute_capability TEXT,
    vram_mb INTEGER,
    host_ram_mb INTEGER,

    status TEXT NOT NULL DEFAULT 'provisioning',
    current_job_id TEXT,

    runtime_image TEXT NOT NULL,
    runtime_image_digest TEXT,
    runtime_build_revision TEXT,

    cuda_version TEXT,
    pytorch_version TEXT,
    comfy_commit TEXT,
    comfy_kitchen_version TEXT,

    loaded_family TEXT,
    loaded_model TEXT,
    loaded_quantization TEXT,
    loaded_text_encoder TEXT,
    loaded_vae TEXT,

    capabilities_json TEXT NOT NULL DEFAULT '{}',
    telemetry_json TEXT NOT NULL DEFAULT '{}',
    warm_state_json TEXT NOT NULL DEFAULT '{}',

    heartbeat_at INTEGER,
    last_progress_at INTEGER,
    last_error_code TEXT,
    last_error TEXT NOT NULL DEFAULT '',
    debug_log_json TEXT NOT NULL DEFAULT '[]',

    provision_deadline_at INTEGER,

    idle_since INTEGER,
    idle_timeout_seconds INTEGER NOT NULL DEFAULT 60,
    terminate_after INTEGER,
    last_used_at INTEGER,

    draining_at INTEGER,
    drain_reason TEXT,

    delete_attempts INTEGER NOT NULL DEFAULT 0,
    delete_next_retry_at INTEGER,
    delete_last_error TEXT NOT NULL DEFAULT '',
    deletion_requested_at INTEGER,
    provider_deleted_at INTEGER,

    created_at INTEGER NOT NULL,
    ready_at INTEGER,
    updated_at INTEGER NOT NULL
);
```

Indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_image_pod_workers_reuse
  ON image_pod_workers(provider, status, service_kind, terminate_after, updated_at);

CREATE INDEX IF NOT EXISTS idx_image_pod_workers_project_status
  ON image_pod_workers(project_id, status, updated_at);

CREATE INDEX IF NOT EXISTS idx_image_pod_workers_provider_instance
  ON image_pod_workers(provider, provider_instance_id);

CREATE INDEX IF NOT EXISTS idx_image_pod_workers_timeout
  ON image_pod_workers(status, terminate_after, heartbeat_at);

CREATE INDEX IF NOT EXISTS idx_image_pod_workers_delete_retry
  ON image_pod_workers(status, delete_next_retry_at, delete_attempts);
```

### Critical ownership rule

`project_id` is **current assignment/observability metadata only**.

When a pod becomes healthy idle:

```text
current_job_id = NULL
project_id = NULL
user_email = NULL
status = idle
provider_detail/warm_state may remain
```

The pod is now global capacity and may immediately accept a compatible queued image from **any project**.

Never reserve an idle Krea/image pod for the project that created it.

---

## 18. Per-project capacity policy: max 5, 1 pod per 3 ready images

Configuration:

```text
IMAGE_POD_MAX_ACTIVE_PER_PROJECT = 5
IMAGE_POD_QUEUED_IMAGES_PER_NEW_POD = 3
IMAGE_POD_CONCURRENCY_PER_WORKER = 1
```

For ready queued images belonging to one project:

```text
1-3 images   -> target 1 pod
4-6 images   -> target 2 pods
7-9 images   -> target 3 pods
10-12 images -> target 4 pods
13+ images   -> target 5 pods max
```

Formula:

```text
desired_project_capacity = min(5, ceil(ready_queued_jobs / 3))
```

The cap counts workers currently **reserved/provisioning/starting/busy for that project**. Global idle workers with `project_id=NULL` do not belong to any project and do not count until atomically claimed for a job.

### Capacity order

Always do this in order:

```text
1. use a compatible ready idle global pod first
2. use already-starting compatible capacity when appropriate
3. only provision new RunPod/Novita instances for the remaining deficit
```

Never create a fresh pod while compatible healthy idle global capacity is sitting available.

For a frontend window of 10 ready images, normal target capacity is:

```text
ceil(10 / 3) = 4 pods
```

Each of those four pods still processes only one image at a time and then immediately grabs another compatible queued image.

---

## 19. Global pool scheduling

Copy the useful H3 global-pool behavior into the generic image dispatcher.

### Immediate post-job handoff

When a pod finishes an image:

```text
1. upload/finalize result
2. clear job-specific GPU state
3. mark worker idle/global
4. immediately attempt an atomic claim of the next compatible queued image
5. if a job exists, mark busy and dispatch without waiting for the next cron
6. if no job exists, start normal idle timeout
```

This makes warm capacity useful across projects immediately.

### Job selection

Primary scheduling order:

```text
compatible task family/runtime
project below 5-active-pod cap
oldest ready queued job first
```

Secondary affinity may prefer, without starving older work:

```text
same loaded_family/model
same VAE
same hot user-LoRA set
```

Affinity is a latency optimization only; it must not turn workers into project-owned or LoRA-owned pods.

### Cold-start reservation recovery

If a ready global worker appears while another fresh instance is still pulling for the same queued job, the ready worker may take the job. The still-pulling instance must **not** be blindly deleted if it can safely become another global idle worker when ready. This mirrors the existing H3 global-pool approach.

---

## 20. Frontend/D1 batching — window of 10, durable jobs are the scheduler source

Current CharacterScreen and Storyboard generation code uses a **maximum concurrency window of 10**. It does not reliably submit one atomic 10-item D1 batch: the browser starts individual image-generation calls, and Storyboard may defer scenes whose previous-image dependency is not ready.

For the image-pod architecture, D1 must remain the durable source of scheduling truth. Do not make pod capacity depend on React promises staying alive.

### Target API behavior

Add/extend a generic batch submission path:

```text
POST /api/image-generation/batch
max items per request = 10
surface = characters | storyboard
```

The Worker should insert the accepted batch and all its image job rows before returning success.

For >10 targets, the frontend sends additional chunks of up to 10. It still does **not** need to send the entire project at once.

### `image_generation_batches`

Add a lightweight generic UI/request grouping table:

```sql
CREATE TABLE IF NOT EXISTS image_generation_batches (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    user_email TEXT,
    surface TEXT NOT NULL,
    model_key TEXT,
    requested_count INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    completed_count INTEGER NOT NULL DEFAULT 0,
    failed_count INTEGER NOT NULL DEFAULT 0,
    cancelled_count INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    completed_at INTEGER,
    updated_at INTEGER NOT NULL
);
```

Extend existing `image_generation_jobs` generically with fields such as:

```text
request_batch_id
batch_index
batch_total
pod_worker_id
requested_provider
actual_provider
settings_json
execution_json / provider_detail_json
attempt_log_json
heartbeat_at
last_progress_at
started_at
```

The existing durable per-image row remains the leaf source of truth for status, billing, cancellation, output and thumbnails.

### Storyboard dependencies

Do not hold a dependent scene only in browser memory.

If a storyboard image needs the previous scene image:

```text
insert its D1 image job now
persist dependency_target_id / dependency job relationship
mark it blocked/not-ready for dispatch
unlock it automatically when the required predecessor completes
```

The pod scheduler counts only **ready queued** jobs when calculating the 1-pod-per-3-images target.

Character jobs are normally independent and can all become ready immediately.

---

## 21. No multi-image pod workload

Because the product rule is **1 pod = 1 active image**, do not create H3-style multi-item image workloads as the main scheduling abstraction.

Use:

```text
image_generation_jobs = durable leaf jobs
image_generation_batches = frontend/request grouping only
image_pod_workers = reusable GPU capacity
```

The pod HTTP endpoint may retain the familiar `/workloads` route for lifecycle parity, but an image workload contains exactly **one image item** in v1.

That keeps cancellation, retry, billing, progress and reassignment simple.

---

## 22. Generic lifecycle auxiliary tables

Use mature Enhancer-style durable coordination.

### Provider deletion lock

```sql
CREATE TABLE IF NOT EXISTS image_pod_delete_locks (
    worker_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    provider_instance_id TEXT,
    owner_token TEXT NOT NULL,
    locked_at INTEGER NOT NULL,
    lease_until INTEGER NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    next_retry_at INTEGER,
    last_error TEXT NOT NULL DEFAULT '',
    provider_deleted_at INTEGER,
    verified_at INTEGER,
    updated_at INTEGER NOT NULL
);
```

### Event nonce / replay protection

```sql
CREATE TABLE IF NOT EXISTS image_pod_event_nonces (
    nonce TEXT PRIMARY KEY,
    worker_id TEXT NOT NULL,
    job_id TEXT,
    event_type TEXT,
    event_timestamp INTEGER NOT NULL,
    received_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL
);
```

### Dispatch lease

Use a short atomic D1 lease/state transition when claiming:

```text
queued job -> worker
idle worker -> job
```

so two concurrent request handlers/cron invocations cannot claim the same job or worker.

A dedicated `image_pod_dispatch_locks` table is acceptable, but an atomic conditional update is preferred if it is simpler and equally safe.

---

## 23. Cron/reconciliation cadence

Reuse SceneBuilder's existing scheduled-trigger structure.

### Every 1 minute — dispatcher + fast reconciler

Use the existing:

```text
* * * * *
```

handler to:

```text
claim queued ready image jobs for idle global pods
calculate per-project desired capacity with ceil(ready/3), max 5
provision only real capacity deficits
recover stale reservations
refresh heartbeat/progress-derived state
release project association from idle workers
requeue jobs whose worker reservation safely failed
check draining workers that are now empty
```

Immediate job-completion handoff remains event-driven; the one-minute cron is a recovery/fill-the-gaps mechanism, not the normal latency path.

### Every 15 minutes — deletion/reaper reconciliation

Reuse the existing 15-minute cadence pattern:

```text
7,22,37,52 * * * *
```

for image-pod provider cleanup/reconciliation:

```text
delete expired/draining idle pods
retry failed RunPod/Novita deletions
verify provider instance actually disappeared
clean stale failed provisioning workers
reconcile D1 rows against provider truth
clean expired event nonces/locks where appropriate
```

Do not rely on the pod process exiting to stop provider billing. SceneBuilder must confirm deletion through the provider APIs.

---

## 24. Generic image pod HTTP/state-machine parity with H3 + Enhancer

HTTP API:

```text
GET  /health
GET  /ready
GET  /diagnostics
GET  /diagnostics/gpu

POST /workloads                  # exactly one image item in v1
GET  /workloads/:id
POST /workloads/:id/cancel
POST /workloads/:id/items/:jobId/cancel
```

Protected endpoints:

```text
Authorization: Bearer <SCENEBUILDER_POD_TOKEN>
```

Auth reuse:

```text
H3_POD_AUTH_MASTER_SECRET
SCENEBUILDER_POD_TOKEN = HMAC(H3_POD_AUTH_MASTER_SECRET, workerId)
```

No Krea-only pod auth secret.

### Queue/busy behavior

- queue capacity 1;
- exactly one active image at a time;
- duplicate workload/job request is idempotently acknowledged;
- busy/queued worker rejects another workload instead of overcommitting VRAM;
- draining worker rejects all new work.

### Worker states

```text
provisioning
pulling
starting
idle
busy
draining
deleting
deleted
unhealthy
failed
```

### Idle timeout

After a completed job, first attempt immediate global handoff. Only if no compatible queued job exists:

```text
status = idle
idle_since = now
terminate_after = now + idle_timeout_seconds
project_id = NULL
user_email = NULL
```

When deadline expires:

```text
status = draining
emit idle_expired
refuse new work
```

Control plane owns provider deletion.

### Draining

Enter draining for:

```text
idle expiration
explicit retirement
runtime/image rollout
fatal CUDA/model state
provider mismatch/unsafe worker
```

Draining rejects new jobs, lets safe active work finish unless cancelled, and is deleted only when no image remains active.

### Provision/readiness timeout

Persist `provision_deadline_at`. If `/ready` does not succeed before deadline:

```text
mark failed/unhealthy
record diagnostics
request provider deletion
requeue image job if retryable
```

### Heartbeat/stale reconciliation

Track:

```text
heartbeat_at
last_progress_at
telemetry_json
```

Recover contradictions such as:

```text
provider instance gone but D1 worker exists
D1 busy but pod idle
D1 idle but pod busy
heartbeat stopped
job progress stalled beyond timeout
provider deleted but row not finalized
```

---

## 25. Generic events and error handling

Events:

```text
worker_ready
worker_unhealthy
workload_accepted
workload_started
job_started
job_progress
job_completed
job_failed
job_cancel_requested
job_cancelled
workload_completed
workload_failed
worker_idle
idle_expired
model_switch
```

Krea phases:

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

Normalized error examples:

```text
CUDA_UNAVAILABLE
CUDA_DRIVER_TOO_OLD
CUDA_OOM
GPU_CAPABILITY_MISMATCH
GPU_VRAM_BELOW_POLICY
MODEL_LOAD_FAILED
TEXT_ENCODER_LOAD_FAILED
VAE_LOAD_FAILED
COMFY_START_FAILED
COMFY_PROMPT_FAILED
COMFY_INTERRUPTED
R2_INPUT_FAILED
R2_LORA_FAILED
R2_OUTPUT_FAILED
INVALID_LORA_ID
LORA_INCOMPATIBLE
INVALID_SETTINGS
STYLE_REFERENCE_FAILED
JOB_TIMEOUT
PROVISION_TIMEOUT
CALLBACK_AUTH_FAILED
PROVIDER_CREATE_FAILED
PROVIDER_DELETE_FAILED
CANCELLED
UNKNOWN
```

Fatal CUDA/model-corruption errors drain the worker. User/settings errors fail only the image and leave a healthy worker reusable.

Never expose secrets in diagnostics/debug logs.

---

## 26. Cancellation and model switching

Active Comfy cancellation uses `/interrupt`.

Cancellation must be idempotent.

Same-family Krea -> Krea:

```text
preserve warm Krea/Qwen/VAE host state
preserve compatible hot user-LoRA tensors where useful
release only job GPU state
```

Future incompatible image-family switch:

```text
emit model_switch
finish/cancel current image
fully release incompatible loaded family
clear GPU allocator
load next adapter family
update loaded_* fields
```

---

## 27. Provider routing and GPU policy

Reuse Enhancer GPU inventory/name normalization.

Krea initial minimum:

```text
20 GB VRAM
```

Eligible RunPod classes are existing catalog entries at or above 20 GB; current Novita candidates include 4090, 5090, RTX 6000 Ada and L40S.

OOM escalation:

```text
20 GB -> 24 GB -> 32 GB -> 48 GB
```

Do not retry the same failed tier indefinitely.

A functional qualification is still required per GPU class to prove the pinned CUDA/PyTorch/Comfy/INT8 stack runs. This is not a visual quality benchmark.

Future image models define their own minimum GPU/capability rules while sharing the same worker/provisioning infrastructure.

---

## 28. Native Krea workflows/manifests

Files:

```text
krea2/workflows/krea2_turbo.json
krea2/workflows/krea2_style_reference.json

krea2/workflows/manifests/krea2_turbo.json
krea2/workflows/manifests/krea2_style_reference.json
```

Normal workflow:

```text
UNETLoader -> Krea 2 Turbo INT8
CLIPLoader -> Qwen3-VL-4B BF16, type=krea2
0-3 resolved user MODEL LoRA patches
positive encode
optional guided negative encode OR native zero-negative path
EmptyLatentImage
KSampler
selected VAE
VAEDecode
output
```

Style-reference workflow:

```text
1-3 style images
baked krea2_style_reference.safetensors
Krea INT8
Qwen3-VL-4B BF16
selected VAE
prompt/settings
native reference conditioning
sampling/decode/output
```

Reproduce the native Comfy reference behavior rather than inventing a separate reference algorithm.

---

## 29. Provider-neutral Krea payload

Example normal LoRA-mode image:

```json
{
  "jobId": "job_123",
  "projectId": "proj_123",
  "taskFamily": "krea2_image",
  "model": "krea-2-turbo",
  "prompt": "...",
  "negativePrompt": "",
  "width": 2048,
  "height": 1152,
  "settings": {
    "styleMode": "lora",
    "resolutionTier": "quality",
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
      { "loraId": "<uuid>", "strength": 0.8 }
    ]
  },
  "inputs": {
    "outputPrefix": "projects/proj_123/images/generated"
  }
}
```

Style-reference example uses:

```text
styleMode = reference_images
loras = []
inputs.styleImages = 1-3 trusted project object keys
```

Worker validates exclusivity and resolves private assets before dispatch.

---

## 30. SceneBuilder2 integration

Use the existing durable image-generation job model and browser status/cancel/finalization flow.

Target flow:

```text
CharacterScreen / Storyboard
  -> image batch submit, <=10 items
  -> D1 image_generation_batches + image_generation_jobs
  -> generic image dispatcher/global pool
  -> image_pod_workers
  -> RunPod/Novita pod, one image at a time
  -> existing image job status/cancel/finalization
  -> project R2 + existing thumbnail flow
```

Do not create a Krea-only polling protocol.

Persist immutable job settings:

```text
model key + quantization
task family
resolution tier + exact width/height
raw prompt
effective/enhanced prompt
negative prompt
seed
steps/cfg/sampler/scheduler/denoise
VAE
text encoder profile
styleMode
ordered loraId + strength list OR style image object keys
requested provider/backend
actual provider/worker/GPU metadata
```

Shared user-LoRA component:

```text
<LoraSelector modelKey="krea-2-turbo" maxSelected={3} />
```

With one or more user LoRAs selected, style-image controls are disabled. With zero user LoRAs, user may send 1-3 style references.

---

## 31. Readiness/diagnostics contract

`/ready` verifies:

```text
GPU visible
minimum VRAM/capability for advertised families
runtime build metadata
Comfy starts/responds
required baked files discoverable
Krea2 model class present
KREA2 CLIP type present
comfy-kitchen present
baked style-reference adapter present
R2 config available for dynamic inputs/outputs/user LoRAs
worker not draining
```

`/diagnostics` safely exposes:

```text
worker ID/status
current job/project assignment
loaded family/model
idle_since/terminate_after
uptime
GPU name/UUID/driver/VRAM/utilization
CUDA/PyTorch/Comfy versions
required-file checks
disk total/used/free
user-LoRA cache bytes/count
host RAM usage
warm-state summary
last error code
```

Never include provider/R2 credentials or bearer tokens.

---

## 32. Functional tests / canaries

These tests verify runtime correctness; they are **not a request to benchmark competing resolutions/settings**.

### Build/runtime

```text
35 GB rootfs policy respected
no network volume configured
CUDA 13 visible
PyTorch 2.13 + cu130
single pinned Comfy installation discovers external model root
comfy-kitchen 0.2.28 present
native Krea2 recognized
Qwen3-VL-4B KREA2 CLIP recognized
baked krea2_style_reference discovered
INT8 ConvRot generates
no Sage/FlashAttention dependency
```

### Resolution smoke

```text
2048x1152
1152x2048
1280x720
720x1280
```

No `1368x768` decision benchmark is required.

### VAE smoke

```text
Qwen Image VAE decodes
Wan 2.1 VAE decodes
```

### Style-reference smoke

```text
1 / 2 / 3 references
square + landscape + portrait
mixed dimensions
odd unaligned dimensions
zero user LoRAs enforced
baked system style adapter auto-used
```

### User-LoRA cache

```text
0 LoRA
MinimalisticVectorArt only
Darkchurch only
both
3 user LoRAs when a third compatible asset exists
same display name / different IDs selects correct file
invalid ID rejected
4 user LoRAs rejected
R2 miss -> local disk cache
same next LoRA -> hot RAM reuse
same LoRA different strength -> same tensors, new patch strength
new LoRA -> old tensors evictable, disk cache retained
LRU/watermark eviction
```

### Global image-pod lifecycle

```text
provisioning -> ready -> idle
idle -> busy -> idle
one pod never executes two images concurrently
busy rejects second workload
same job/workload request is idempotent
idle worker clears project ownership
idle global worker takes a queued image from another project
project never exceeds 5 reserved/provisioning/starting/busy pods
1-3 ready images targets 1 pod
4-6 targets 2
7-9 targets 3
10 targets 4
13+ caps at 5
compatible idle global capacity is used before new provisioning
job completion immediately attempts next global claim
draining rejects new work
idle expiration -> draining -> provider delete
lost callback recovered by cron/reaper
provider deletion retry/lock
heartbeat/provision/job timeout recovery
queued cancel
active /interrupt cancel
pod crash during generation
fatal CUDA error drains worker
normal validation error leaves worker reusable
runtime rollout drains old workers
future incompatible family performs full model switch
```

### Batch durability

```text
Character batch of <=10 persists all accepted image rows
Storyboard batch of <=10 persists all accepted rows
blocked previous-scene dependency is in D1, not browser-only
blocked job unlocks after predecessor succeeds
browser refresh/close does not lose already accepted jobs
multiple <=10 chunks work for >10 requested targets
scheduler counts only ready queued jobs for capacity
```

---

## 33. Rollout order

### Phase 1 — base Krea runtime

1. Build CUDA 13/PyTorch 2.13 base.
2. Bake Krea INT8 -> VAEs -> Qwen -> `krea2_style_reference.safetensors` under `/opt/scenebuilder-models/krea2`.
3. Install one pinned Comfy layer afterward and configure external model paths.
4. Add nodes -> workflows -> final generic image runtime.
5. Verify 35 GB disk headroom.
6. Reproduce native Turbo defaults.
7. Smoke-test locked quality + fast resolutions.
8. Pass 20 GB functional CPU-offload canary.

### Phase 2 — generic image-pod lifecycle

1. Implement generic image pod server/adapter registry.
2. Enforce one active image per pod.
3. Add H3/Enhancer busy/draining/idle/readiness/diagnostics/cancellation behavior.
4. Add normalized errors, heartbeat and callbacks.
5. Add RunPod + Novita provisioning with 35 GB root disk/no network volume.

### Phase 3 — D1 image worker/global pool

1. Add `image_pod_workers`.
2. Add delete locks/event nonce protection/dispatch lease.
3. Implement global idle ownership clearing.
4. Implement immediate cross-project warm handoff.
5. Implement 5-pod project cap and `ceil(ready_jobs/3)` desired capacity.
6. Add one-minute dispatcher/reconciler and fifteen-minute deletion/reaper handler.

### Phase 4 — durable frontend batches

1. Add `image_generation_batches`.
2. Add batch metadata to `image_generation_jobs`.
3. Accept max 10 items per browser batch request.
4. Persist Storyboard dependency-blocked jobs immediately.
5. Make D1, not React concurrency, the scheduler source of truth.

### Phase 5 — warm memory lifecycle

1. Keep Krea/Qwen/VAE reusable CPU states warm.
2. Clear job GPU allocations without automatic full model unload.
3. Add host-memory pressure eviction.
4. Track warm-state telemetry.

### Phase 6 — user-LoRA catalog/cache

1. Add generic `lora` + `lora_model_support`.
2. Seed the existing two R2 user LoRAs with immutable IDs.
3. Implement trusted R2 -> local disk cache.
4. Implement hot parsed-LoRA RAM reuse.
5. Add disk LRU/watermarks.

### Phase 7 — style-reference mode

1. Reproduce native workflow with baked `krea2_style_reference.safetensors`.
2. Support 1-3 mixed-size references when no user LoRA is selected.

### Phase 8 — SceneBuilder UI

1. Add `krea-2-turbo` to image model selection.
2. Use quality `2048x1152` / portrait equivalent as normal default.
3. Offer fast `1280x720` / portrait equivalent if desired.
4. Add D1-ID user-LoRA picker.
5. Add LoRA-or-style-reference controls.
6. Add advanced generation controls.

---

## 34. Guardrails

- No network volume/network disk.
- Provider root/container disk is 35 GB.
- Krea/Qwen/VAEs and `krea2_style_reference.safetensors` are baked.
- User-selectable LoRAs are dynamic R2/D1 assets and are not baked.
- Dynamic LoRA local disk cache is allowed and bounded.
- D1 `lora_id` is user-LoRA API/cache identity; display names may collide.
- Browser never controls arbitrary LoRA filesystem/R2 paths.
- Maximum 3 user LoRAs.
- Style-reference mode uses zero user LoRAs in v1 and automatically uses the baked system adapter.
- Style images may use different dimensions/aspects.
- Quality default is 2048x1152 / 1152x2048; do not use 1368x768 as the normal preset.
- Keep Krea/Qwen warm in CPU RAM where safe.
- Normal post-job cleanup frees GPU job state without blindly unloading host model state.
- Native Turbo default path remains 8 steps / Euler / simple / denoise1 with no-guidance behavior.
- No SageAttention or FlashAttention dependency in v1.
- Use one pinned Comfy installation; no H3-style two-Comfy retrofit.
- Heavy model layers use a Comfy-independent model root so installing Comfy afterward is safe.
- Generic D1 worker table is `image_pod_workers`, never `krea2_pod_workers`.
- Image pods are global reusable capacity, not project-owned capacity.
- One pod executes one image at a time.
- Maximum 5 active/reserved image pods per project.
- Desired new capacity is one pod per three ready queued images, after accounting for compatible global idle capacity.
- Idle workers clear project/user assignment and immediately look for work from any project.
- Frontend/D1 batch size is max 10 items per request/chunk; D1 durable rows drive scheduling.
- One-minute cron handles dispatch/reconciliation fallback.
- Fifteen-minute cron handles provider deletion/reaper reconciliation.
- Reuse H3 pod auth master secret; no Krea-only secret.
- Preserve the existing per-image status/cancel/finalization semantics.

---

## 35. First implementation milestone

```text
35 GB container disk only
no network storage
CUDA 13 / PyTorch 2.13 cu130
Krea 2 Turbo INT8 ConvRot
Qwen3-VL-4B BF16
Qwen Image + Wan 2.1 VAE baked
krea2_style_reference.safetensors baked
weights under /opt/scenebuilder-models/krea2
one pinned Comfy installation after heavy weight layers
extra Comfy model paths configured
nodes -> workflow -> generic runtime last
no Sage/FlashAttention
quality default 2048x1152 / 1152x2048
optional fast 1280x720 / 720x1280
8 steps / Euler / simple / denoise1 native Turbo defaults
one RunPod >=20 GB functional canary
warm Krea/Qwen CPU offload proven
one image at a time per pod
```

Acceptance criteria:

```text
Comfy boots and discovers pre-baked external model root
native Krea2 + KREA2 Qwen CLIP load
baked style-reference adapter is discoverable
BF16 encoder works
Krea/Qwen move between GPU and CPU/offload state on a 20 GB worker
job GPU allocations clear after completion
warm next job avoids unnecessary checkpoint reload
both VAEs decode
quality + fast aligned presets generate
seed/steps/CFG/sampler/scheduler/denoise are runtime-patchable
output uploads to R2
pod remains reusable
busy/draining/idle behavior works
generic image worker identity is used
```

Then implement D1 global-pool scheduling/batching, dynamic user-LoRA catalog/cache, and SceneBuilder UI rollout.
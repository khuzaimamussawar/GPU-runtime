# SceneBuilder Krea 2 Runtime Plan

Status: **architecture / implementation plan**

Scope: add Krea 2 Turbo image generation to SceneBuilder Storyboard and CharacterScreen, while building a **generic image-generation pod/control-plane layer** that can host more image models later. Krea 2 is the first image family, not the name or boundary of the pod infrastructure.

This file is the source of truth for the Krea 2 implementation and the reusable image-pod architecture around it.

---

## 1. Locked product decisions

### Krea 2 model/runtime

- Primary model: **Krea 2 Turbo INT8 ConvRot**.
- Diffusion checkpoint: `krea2_turbo_int8_convrot.safetensors`.
- Runtime: **native ComfyUI Krea 2 support**.
- Text encoder: **Qwen3-VL-4B BF16** (`qwen3vl_4b_bf16.safetensors`).
- Qwen3-VL-4B is used for text conditioning and native image/style-reference conditioning.
- Minimum GPU: **20 GB VRAM**.
- One active generation at a time per GPU for v1.
- Providers: **RunPod Pods + Novita GPU instances**.
- Reuse the same RunPod/Novita/R2 variables and H3 pod-auth master secret already used by the existing GPU control plane.

### Storage

- **No network volume.**
- **No network disk.**
- **No provider-mounted model storage.**
- Provider root/container disk: **35 GB**.
- Krea model, Qwen encoder and VAEs are baked into Docker image layers.
- User/style LoRAs are not baked into Docker.
- LoRAs may be downloaded from R2 and cached on the 35 GB local container disk.
- Generated inputs/outputs may use temporary local files during a job, then are cleaned after upload/finalization.

Provider configuration:

```text
IMAGE_POD_DISK_GB = 35
RunPod: containerDiskInGb = 35
Novita: rootfsSize = 35
Novita: networkStorages = []
RunPod: no network volume / volume mount
```

The image-size gate must leave enough writable space for:

```text
Comfy input/output/temp
atomic LoRA downloads
bounded LoRA cache
logs / small runtime scratch
```

Use measured free-space watermarks rather than assuming a fixed LoRA-cache capacity before the final image size is known.

### VAE

User-selectable:

```text
qwen_image -> qwen_image_vae.safetensors
wan_2_1    -> selected Wan 2.1 VAE checkpoint
```

Rules:

- `qwen_image` is the default/native Krea 2 path.
- `wan_2_1` remains selectable only after matched-seed runtime validation.
- Qwen text conditioning is independent of the selected VAE decode path.

### LoRAs

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

- Every LoRA gets an immutable D1 `lora_id`.
- Browser/API/job identity is `loraId`, never display name or filename.
- Display names are intentionally not unique.
- D1 resolves `lora_id -> trusted exact R2 object key` before dispatch.
- Optional SHA-256/ETag metadata is integrity/version metadata, not identity.

---

## 2. Native ComfyUI baseline

Use the same tested Comfy revision currently used by H3 unless a Krea-specific canary proves that a deliberate upgrade is required:

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

## 3. CUDA / PyTorch / Python baseline

Start with the same proven software family as H3:

```text
CUDA: 13.0 / cu130
PyTorch: 2.13.0
Python: 3.13 target
Ubuntu: 24.04
```

The pinned Comfy quantization path enables optimized `comfy-kitchen` CUDA operations on CUDA 13+.

Build stages may use CUDA devel when required. Prefer a matching CUDA 13 runtime image for the final stage if the real ConvRot canary passes.

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

## 4. Docker layer order and why Comfy can come after the heavy weights

It is safe for the **Docker build lineage** to add Krea/Qwen/VAE files before installing Comfy. Docker layer order is not runtime execution order: the final container filesystem contains every lower layer before the runtime starts.

The only thing that would be unsafe is placing model files inside `/opt/ComfyUI` before cloning/installing Comfy into that same non-empty directory.

Therefore heavy weight layers must use a Comfy-independent model root:

```text
/opt/scenebuilder-models/krea2/
  diffusion_models/
    krea2_turbo_int8_convrot.safetensors
  text_encoders/
    qwen3vl_4b_bf16.safetensors
  vae/
    qwen_image_vae.safetensors
    <wan-2.1-vae>.safetensors
```

Then the single Comfy layer installs into:

```text
/opt/ComfyUI
```

and wires the baked SceneBuilder model root using Comfy `extra_model_paths.yaml` (or equivalent stable folder-path registration) so native loaders see the files in the normal categories.

This means **we do not need H3's historical two-Comfy/core-overlay retrofit**.

Recommended linear chain:

```text
00 image-krea2-base
   CUDA 13 + Python + PyTorch 2.13/cu130 + common runtime packages

10 image-krea2-model-int8
   add Krea 2 Turbo INT8 under /opt/scenebuilder-models/krea2

20 image-krea2-vaes
   add Qwen Image + selected Wan 2.1 VAE

30 image-krea2-qwen-bf16
   add Qwen3-VL-4B BF16

40 image-krea2-comfyui
   install the single pinned Comfy revision into /opt/ComfyUI
   install matching requirements/comfy-kitchen
   configure extra model paths to /opt/scenebuilder-models/krea2
   verify Krea2 + KREA2 CLIP + ConvRot discovery

50 image-krea2-nodes
   only custom/helper nodes actually required

60 image-krea2-workflow
   API workflows + manifests

70 image-krea2-runtime
   LAST layer
   generic image pod HTTP server
   model-family adapter registry
   R2 + output handling
   LoRA cache manager
   workflow patching
   cancellation/progress/readiness
   memory cleanup/offload
   idle/draining lifecycle
```

Consequences:

- changing Comfy does not redownload the parent Krea/Qwen/VAE layers;
- changing nodes rebuilds nodes/workflow/runtime only;
- changing workflows rebuilds workflow/runtime only;
- changing runtime rebuilds only the last layer;
- adding LoRAs changes D1/R2 only.

The Krea final image can still be named specifically, for example:

```text
khuxaima/scenebuilder-krea2-pod:latest
```

while the **control-plane table and pod protocol remain generic image infrastructure**.

---

## 5. Hetzner layer-by-layer build

Add a dedicated Krea build workflow:

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

Push each successful parent so later retries reuse published heavyweight layers.

---

## 6. Resolution contract

Krea 2 is not restricted to one fixed image shape. Use exact SceneBuilder aspect-ratio presets aligned to a 16-pixel grid.

Initial normal presets:

```text
1K landscape: 1280 x 720
1K portrait:   720 x 1280

2K landscape: 2048 x 1152
2K portrait:  1152 x 2048
```

Why not `1368x768` as the normal product default:

- it is a ~1 MP calculator result, not a special Krea training resolution;
- `1368` is not divisible by 16;
- the model can pad it, but SceneBuilder gets no benefit from hidden padding;
- `1280x720` is exact 16:9, fully aligned and cheaper on 20 GB GPUs.

Canary `1368x768` once against `1280x720`. If the larger pixel count gives a meaningful quality gain, compare an aligned exact-ratio alternative such as `1536x864` rather than making a slightly misaligned size the default.

Optional 2K performance fallback to benchmark:

```text
1792 x 1008
1008 x 1792
```

Do not silently downgrade a requested 2K job; use GPU routing/escalation policy.

---

## 7. Sampling/settings contract

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

These are defaults, not hidden constants. User-controllable fields:

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
strength for each selected LoRA
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

Native Turbo default uses zeroed negative conditioning. At `CFG=1`, a typed negative prompt is effectively inactive.

UI/runtime behavior:

- CFG 1: show negative prompt as inactive/no effect;
- CFG > 1: encode supplied negative text and feed it as guided negative conditioning;
- validate higher-CFG behavior separately because Turbo is distilled for low/no guidance.

### Prompt enhancement

Always store the raw user prompt. If enhancement is enabled, also store the effective/enhanced prompt in execution metadata so regeneration is reproducible.

---

## 8. Two style paths

Per generation choose one of:

```text
A. user LoRA mode
   1-3 user-selected D1 LoRA IDs
   no style-reference images

B. style-reference mode
   zero user-selected LoRAs
   1-3 style-reference images
   hidden/internal Krea style-reference adapter/workflow
```

Do not combine arbitrary user LoRAs and style-reference images in v1.

The native Krea style-reference workflow uses an internal `krea2_style_reference.safetensors` adapter. Treat it as a hidden/system LoRA:

- not visible in normal UI LoRA selection;
- stored in R2, not baked into Docker;
- represented by an internal D1 LoRA row (`is_internal=1`) or an equivalent trusted system-asset row;
- cached locally like other LoRAs;
- automatically selected by style-reference mode.

---

## 9. Style-reference image dimensions

Style references do **not** need to match:

```text
each other
output width/height
output aspect ratio
```

Do not stretch all references to the output shape.

Runtime preprocessing policy:

```text
1. EXIF-orient
2. convert to RGB
3. preserve source aspect ratio
4. do not force the output aspect ratio
5. only downscale inputs that are too large for the qualified memory policy
6. let pinned native Qwen/Krea preprocessing align each image to its own patch grid
```

Canary mixed references:

```text
1024x1024
1280x720
720x1280
odd non-16/32-aligned user sizes
all three together
```

---

## 10. Generic D1 LoRA catalog

Create one catalog that can serve future image and video models.

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

Rules:

- `id` is the immutable API/cache/job identity.
- `display_name` is not unique.
- `thumbnail_object_key` is UI-only.
- `lora_type`: `image | video | both`.
- `is_internal=1` hides system adapters from normal UI catalogs.

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

The same physical R2 file can support multiple future model keys/quantizations without duplication.

---

## 11. LoRA local disk + hot RAM policy

Local cache path:

```text
/opt/scenebuilder-image/cache/loras/<lora_id>.safetensors
```

Use immutable `lora_id`, never display name, as the filename/cache key.

First use:

```text
browser loraId
 -> Worker resolves D1 asset + compatibility
 -> pod receives trusted loraId + R2 object key + optional integrity metadata
 -> cache hit: reuse local file
 -> cache miss: R2 -> <lora_id>.part -> validate -> atomic rename
 -> parse LoRA tensors
 -> apply model patch at requested strength
```

### Hot CPU cache

Keep the currently useful parsed LoRA state/tensors in CPU RAM when memory allows.

Cache identity should include asset version when available:

```text
lora_id + etag/version/sha256
```

Strength is **not** part of cache identity; the same parsed LoRA can be repatched with a different strength.

If the next job uses the same LoRA set:

```text
reuse hot CPU tensors
avoid disk/R2 read
repatch at requested strengths
```

If the next job uses a different set:

```text
release old hot tensors as needed
leave old safetensors in bounded disk cache
load new set from disk cache or R2
```

Disk eviction:

- never evict baked model/Qwen/VAE assets;
- never evict an active job's LoRA;
- delete oldest unused LoRA cache files first;
- redownload later from R2 when needed.

---

## 12. Krea/Qwen/VAE warm CPU offload policy

The main CPU-offload target is the expensive **Krea model + Qwen text encoder**, not the LoRA file itself.

Desired post-job state:

```text
VRAM:
  job activations/workspaces/sampler allocations released
  reclaimable CUDA cache cleared

CPU RAM:
  Krea model object/weights retained where safe
  Qwen encoder object/weights retained where safe
  VAE object retained where useful
  current LoRA tensors retained if within hot-cache budget

Container disk:
  baked checkpoints remain
  bounded LoRA safetensors cache remains
```

Constrained-GPU sequence:

```text
1. get Qwen from warm CPU state or load once from baked checkpoint
2. move required encoder state to GPU
3. encode prompt + optional style images
4. offload Qwen back to CPU
5. free reclaimable GPU cache
6. get Krea from warm CPU state or load once
7. activate Krea under Comfy memory management
8. get required LoRAs from hot RAM / disk / R2
9. patch model
10. sample
11. activate selected VAE and decode
12. upload result
13. delete job temp media
14. retain reusable model objects in CPU RAM
15. retain useful LoRA tensors under RAM cap
16. release job-specific GPU memory
```

Do **not** call an end-of-job cleanup equivalent to `unload_models=true` for every job. That defeats warm reuse.

Normal same-family idle cleanup should free reclaimable GPU memory while preserving reusable host-side/offloaded model state. A full model unload is appropriate for:

```text
model-family switch
fatal CUDA state
host-RAM pressure that requires eviction
pod draining/deletion
```

Host-RAM pressure eviction priority:

```text
old hot LoRA tensors
inactive VAE
Qwen warm state
Krea warm state last
```

---

## 13. Enhancer engine pattern to copy

Enhancer already implements the conceptual pattern:

```text
persistent _ENGINE_CACHE
  deserialized reusable engine survives between jobs

job/use _EXECUTION_CACHE
  execution context/CUDA stream/workspace is releasable
```

At job end Enhancer releases execution resources and CUDA/CuPy allocator blocks but keeps the process-level engine cache.

Generic image-pod analogue:

```text
warm/process state:
  model-family adapters
  Krea/Qwen/VAE CPU model objects
  current LoRA parsed states
  reusable workflow/model metadata

job-specific state:
  activations
  latents
  sampler tensors
  temporary model patches
  uploaded reference tensors
  CUDA workspace/cache
```

Release job state, preserve useful warm state.

---

## 14. Generic image pod — not Krea-specific

The worker/pod infrastructure must be named and modeled generically because future image families will be added.

Use a generic service identity such as:

```text
scene-builder-image-pod
```

Krea is an adapter/task family:

```text
krea2_image
```

Future examples can be added without a new worker table/protocol:

```text
z_image_turbo
future_krea_base
future_flux_family
other image families
```

A runtime image may advertise one or more supported model/task families in `capabilities_json`. The control plane only reuses a warm worker when the requested job is compatible with that worker image/capabilities.

### Model-family adapter registry

Runtime code should route through a registry rather than hard-coded Krea branches everywhere:

```text
TASK_FAMILY_REGISTRY = {
  "krea2_image": Krea2Adapter,
  ...future image adapters...
}
```

Each adapter owns:

```text
workflow + manifest selection
model files required
request validation
settings patching
style/reference handling
model-switch cleanup rules
result parsing
```

The HTTP server, auth, queue, idle timeout, draining, callbacks, diagnostics and error lifecycle stay generic.

---

## 15. Generic D1 `image_pod_workers` table

Create a **new image-worker table**, independent from Krea naming.

Do not reuse `h3_pod_workers` or `enhancer_pod_workers`, and do not create `krea2_pod_workers`.

Recommended primary schema:

```sql
CREATE TABLE IF NOT EXISTS image_pod_workers (
    id TEXT PRIMARY KEY,

    service_kind TEXT NOT NULL DEFAULT 'image_generation',
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
    current_workload_id TEXT,
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

### Worker states

Use explicit generic states, for example:

```text
provisioning
starting
idle
busy
draining
deleting
deleted
unhealthy
failed
```

`draining` means:

- do not accept new workloads;
- active work may finish or be cancelled;
- once no work remains, control plane deletes/reconciles the provider instance.

### `loaded_*` fields

These are observability/reuse hints, not the source of truth for the model files. Example:

```text
loaded_family = krea2_image
loaded_model = krea-2-turbo
loaded_quantization = int8_convrot
loaded_text_encoder = qwen3vl_4b_bf16
loaded_vae = qwen_image
```

Future image models reuse the same columns.

---

## 16. Generic image pod workload grouping

Keep the existing `image_generation_jobs` table as the durable per-image job source of truth.

Optionally add a lightweight generic dispatch grouping table so Storyboard batches can send several image jobs sequentially to one warm worker without changing per-image ownership/status:

```sql
CREATE TABLE IF NOT EXISTS image_pod_workloads (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    user_email TEXT,

    requested_provider TEXT,
    actual_provider TEXT,
    status TEXT NOT NULL DEFAULT 'queued',

    job_ids_json TEXT NOT NULL DEFAULT '[]',
    task_families_json TEXT NOT NULL DEFAULT '[]',

    pod_worker_id TEXT,
    provider_task_id TEXT,
    gpu_class TEXT,
    region TEXT,

    runtime_image TEXT,
    runtime_image_digest TEXT,

    continue_on_item_failure INTEGER NOT NULL DEFAULT 1,
    idle_timeout_seconds INTEGER NOT NULL DEFAULT 60,

    attempt_log_json TEXT NOT NULL DEFAULT '[]',
    error_code TEXT,
    error TEXT NOT NULL DEFAULT '',

    created_at INTEGER NOT NULL,
    started_at INTEGER,
    completed_at INTEGER,
    updated_at INTEGER NOT NULL
);
```

This table is a dispatch/lifecycle grouping only. `image_generation_jobs` remains the durable leaf-job record used by CharacterScreen/Storyboard status, billing, cancellation and output finalization.

---

## 17. Generic lifecycle auxiliary tables

For parity with the mature Enhancer control plane, add image-specific lifecycle helpers rather than putting provider deletion/replay coordination in memory only.

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

### Event nonce / callback replay protection

```sql
CREATE TABLE IF NOT EXISTS image_pod_event_nonces (
    nonce TEXT PRIMARY KEY,
    worker_id TEXT NOT NULL,
    workload_id TEXT,
    job_id TEXT,
    event_type TEXT,
    event_timestamp INTEGER NOT NULL,
    received_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL
);
```

### Dispatch lease/lock

Use a short D1 lease when assigning a queued image job/workload to a reusable worker so two concurrent dispatcher invocations cannot claim the same `idle` worker.

This can be a dedicated `image_pod_dispatch_locks` table or an atomic conditional state update, but the behavior must be tested explicitly.

---

## 18. Generic image pod HTTP/state-machine parity with H3 + Enhancer

The image pod should combine the strongest mature behavior from H3 and Enhancer.

### HTTP API

```text
GET  /health
GET  /ready
GET  /diagnostics
GET  /diagnostics/gpu

POST /workloads
GET  /workloads/:id
POST /workloads/:id/cancel
POST /workloads/:id/items/:jobId/cancel
```

Protected endpoints use:

```text
Authorization: Bearer <SCENEBUILDER_POD_TOKEN>
```

### Auth

Reuse:

```text
H3_POD_AUTH_MASTER_SECRET
```

Derive per-worker:

```text
SCENEBUILDER_POD_TOKEN = HMAC(H3_POD_AUTH_MASTER_SECRET, workerId)
```

Do not create a Krea-only pod auth secret.

### Queue/busy behavior

- queue capacity 1 for v1;
- one active GPU job at a time;
- duplicate workload ID is idempotently acknowledged;
- if worker is already busy/queued, reject another workload with conflict instead of silently overcommitting VRAM;
- if worker is draining, reject all new workloads.

### Worker transitions

```text
starting
  -> idle/ready
  -> busy
  -> idle
  -> draining
  -> deleting/deleted
```

Failure paths can enter:

```text
unhealthy
failed
```

Control plane owns provider deletion; the pod should never assume that a self-exit guarantees the RunPod/Novita instance was actually deleted/billing stopped.

### Idle timeout

After workload completion:

```text
worker_status = idle
idle_since = now
terminate_after = now + idle_timeout_seconds
```

When deadline expires:

```text
worker_status = draining
emit idle_expired
refuse new work
```

The SceneBuilder control plane performs provider deletion. A D1/provider reaper must recover if the callback is lost.

### Draining

Set draining when:

```text
idle timeout expires
control plane explicitly requests retirement
a runtime/image version is being rolled out
a worker becomes unsafe for reuse after a fatal error
```

Draining semantics:

- reject new jobs;
- allow current job to finish unless cancellation is requested;
- emit state change;
- delete only after no active work remains.

### Provision/readiness timeout

Persist `provision_deadline_at`. If `/ready` does not succeed before deadline:

```text
mark worker failed/unhealthy
record provider/runtime diagnostics
request provider deletion
retry job on another candidate according to job retry policy
```

### Heartbeat/stale-worker handling

Persist/update:

```text
heartbeat_at
last_progress_at
telemetry_json
```

Reaper detects:

```text
provider instance disappeared
pod stops heartbeating
job has no progress beyond timeout
D1 says busy but pod says idle
D1 says worker exists but provider says deleted
```

Reconcile rather than leaving zombie rows/jobs.

---

## 19. Generic events and error handling

Pod callbacks should cover:

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

Recommended Krea progress phases:

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

### Error classes

Normalize errors so control-plane retry/routing can distinguish them, for example:

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

Fatal CUDA/model-corruption errors should mark a worker unsafe for reuse and move it to draining/deletion. A normal user/settings error should fail only the job and leave a healthy worker reusable.

Store:

```text
last_error_code
last_error
debug_log_json
attempt_log_json on job/workload
```

Do not expose raw secrets in debug logs.

---

## 20. Cancellation and model switching

Cancellation of an active Comfy generation uses `/interrupt`.

Job/workload cancellation must be idempotent.

For a same-family Krea job after Krea:

- preserve warm CPU/offloaded state;
- clear job allocations only.

For a future image model family switch:

```text
emit model_switch
finish/cancel current work
fully release incompatible loaded models
clear GPU allocator
load next adapter family
update loaded_family/model fields
```

This mirrors H3's family-switch concept but the image pod registry is generic.

---

## 21. Generic provider routing and GPU policy

Reuse Enhancer's GPU inventory/normalization rather than maintaining a completely separate provider GPU-name universe.

Krea initial rule:

```text
minimum VRAM = 20 GB
```

Initial eligible classes include the existing >=20 GB RunPod catalog and Novita 4090/5090/RTX 6000 Ada/L40S entries.

Every GPU class still needs a real Krea qualification/canary because INT8 ConvRot behavior varies by architecture.

OOM escalation:

```text
20 GB -> 24 GB -> 32 GB -> 48 GB
```

Do not retry the same failed tier indefinitely.

Future image models can define their own minimum VRAM/capability requirements through the adapter/model registry while sharing the same `image_pod_workers` table and provisioning code.

---

## 22. Native Krea workflows/manifests

Keep explicit Krea workflows under the Krea adapter:

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
0-3 resolved MODEL LoRA patches
positive encode
optional guided negative encode OR native ConditioningZeroOut
EmptyLatentImage
KSampler
selected VAE
VAEDecode
output
```

Style-reference workflow:

```text
1-3 style images
internal Krea style-reference adapter
Krea INT8
Qwen3-VL-4B BF16
selected VAE
prompt/settings
native reference conditioning
sampling/decode/output
```

Do not rewrite native reference behavior until the official graph is reproduced successfully.

---

## 23. Provider-neutral Krea payloads

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
      { "loraId": "<uuid>", "strength": 0.8 }
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

Worker validates style-mode exclusivity and resolves all trusted private assets before pod dispatch.

---

## 24. SceneBuilder2 integration

Reuse the existing browser image-generation flow:

```text
CharacterScreen / Storyboard
  -> apiGenerateImage()
  -> /api/generate-image
  -> image_generation_jobs
  -> generic image-pod dispatcher
  -> image_pod_workers / optional image_pod_workloads
  -> RunPod/Novita pod
  -> existing status polling
  -> existing cancellation
  -> R2 finalization
  -> existing thumbnail flow
```

Do not create a Krea-only browser polling protocol.

The immutable image-job settings snapshot should include:

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
ordered loraId + strength list OR style-image object keys
requested provider/backend
actual provider/worker/gpu execution metadata
```

Shared LoRA UI:

```text
<LoraSelector modelKey="krea-2-turbo" maxSelected={3} />
```

When at least one user LoRA is selected, style-image controls are disabled in v1. With zero user LoRAs, user can choose 1-3 style images.

---

## 25. Readiness/diagnostics contract

`/ready` should verify at minimum:

```text
GPU visible
minimum VRAM/capability for advertised image families
runtime image build metadata available
Comfy starts/responds
required Krea baked model files discoverable through Comfy paths
Krea2 model class present
KREA2 CLIP type present
comfy-kitchen backend present
R2 config available when required
worker not draining
```

`/diagnostics` should expose safe operational metadata:

```text
worker ID/status
current workload/job
loaded family/model
idle_since/terminate_after
uptime
GPU name/UUID/driver/VRAM/utilization
CUDA/PyTorch/Comfy versions
required-file checks
disk total/used/free
LoRA cache bytes/count
host RAM usage
warm-state summary
last error code
```

Never include provider/R2 secrets or bearer tokens.

---

## 26. Tests / canaries

### Build/runtime

```text
35 GB rootfs policy respected
no network volume configured
CUDA 13 visible
PyTorch 2.13 + cu130
single pinned Comfy layer discovers external /opt/scenebuilder-models/krea2 paths
comfy-kitchen 0.2.28 present
native Krea2 recognized
Qwen3-VL-4B KREA2 CLIP recognized
ConvRot works
no Sage/FlashAttention dependency
```

### Resolution

```text
1280x720
720x1280
1368x768 comparison only
2048x1152
1152x2048
1792x1008 performance comparison
```

Record quality, peak VRAM, host RAM, time and internal padding.

### VAE

Matched prompt/seed/settings:

```text
Qwen Image VAE
Wan 2.1 VAE
```

Compare correctness, color/detail/artifacts, decode time and memory.

### Style references

```text
1 / 2 / 3 images
square + landscape + portrait
mixed dimensions
odd unaligned dimensions
zero user LoRAs enforced
hidden style adapter auto-selected
```

### LoRA cache

```text
0 LoRA
MinimalisticVectorArt only
Darkchurch only
both
3-LoRA canary when third asset exists
same display name / different D1 IDs selects correct file
invalid ID rejected
4 user LoRAs rejected
R2 miss -> disk cache
next same-LoRA job -> hot RAM reuse
same LoRA different strength -> same tensors, new patch strength
new LoRA -> old tensors evictable, disk cache retained
LRU/watermark eviction
```

### Warm memory

Verify after each job:

```text
job-specific GPU allocations are released
Krea/Qwen do not unnecessarily reload from disk
warm CPU states survive healthy idle
same LoRA hot tensors can be reused
host-memory pressure can evict warm states safely
```

### Generic image-pod lifecycle

Test all of:

```text
provisioning -> ready -> idle
idle -> busy -> idle
busy worker rejects second workload
same workload ID is idempotent
draining worker rejects new work
idle timeout -> draining -> provider delete
lost idle_expired callback recovered by reaper
provider deletion retry + lock
provider already deleted reconciliation
heartbeat timeout
provision timeout
stale busy worker reconciliation
queued cancel
active /interrupt cancel
callback retry/replay protection
pod crash during generation
fatal CUDA error drains worker
normal user validation error leaves worker reusable
runtime version rollout drains old workers
warm worker reuse across multiple image_generation_jobs
future incompatible model family triggers full model switch
```

### Provider matrix

At minimum:

```text
RunPod >=20 GB qualified class
RunPod 4090
RunPod 5090
Novita 4090
Novita 5090
48 GB fallback class
```

---

## 27. Rollout order

### Phase 1 — base Krea runtime

1. Build CUDA 13/PyTorch 2.13 base.
2. Bake Krea INT8 -> VAEs -> Qwen under `/opt/scenebuilder-models/krea2`.
3. Install one pinned Comfy layer afterward and configure external model paths.
4. Add nodes -> workflow -> final generic image runtime.
5. Verify 35 GB disk headroom.
6. Reproduce native Turbo defaults.
7. Canary 1K/2K aligned resolutions.
8. Pass 20 GB CPU-offload canary.

### Phase 2 — generic image-pod lifecycle

1. Implement generic image pod server/adapter registry.
2. Add busy/draining/idle timeout/readiness/diagnostics/cancellation parity with H3/Enhancer.
3. Add normalized errors, heartbeats and callbacks.
4. Add RunPod + Novita provisioning with 35 GB root disk/no network volume.
5. Implement control-plane reaper and safe deletion retries.

### Phase 3 — D1 image worker tables

1. Add `image_pod_workers`.
2. Add `image_pod_delete_locks` and event nonce/replay protection.
3. Add dispatch lease/locking.
4. Add optional `image_pod_workloads` for warm multi-job batching.
5. Keep `image_generation_jobs` as durable leaf jobs.

### Phase 4 — warm memory lifecycle

1. Keep Krea/Qwen/VAE reusable CPU states warm.
2. Clear job GPU allocations without automatic full model unload.
3. Add host-memory pressure eviction.
4. Track warm-state telemetry in `image_pod_workers`.

### Phase 5 — D1 LoRA catalog/cache

1. Add generic `lora` + `lora_model_support`.
2. Seed existing two Krea LoRAs with immutable IDs.
3. Implement trusted R2 -> local disk cache.
4. Implement hot parsed-LoRA RAM reuse.
5. Add disk LRU/watermarks.

### Phase 6 — style-reference mode

1. Register hidden internal Krea style-reference adapter.
2. Reproduce native reference workflow.
3. Support 1-3 mixed-size references when no user LoRA is selected.

### Phase 7 — SceneBuilder UI/control plane

1. Add `krea-2-turbo` to current image model selection.
2. Route GPU-backed image jobs through generic image-pod dispatcher.
3. Preserve existing image job polling/cancel/finalization.
4. Add D1-ID LoRA picker.
5. Add LoRA-or-style-reference controls.
6. Add advanced generation controls.

---

## 28. Guardrails

- No network volume/network disk.
- Provider root/container disk is 35 GB.
- Krea/Qwen/VAE are baked; user/style LoRAs are not.
- LoRA local disk cache is allowed and bounded.
- D1 `lora_id` is LoRA API/cache identity; display names may collide.
- Browser never controls arbitrary LoRA filesystem/R2 paths.
- Maximum 3 user LoRAs.
- Style-reference mode uses zero user LoRAs in v1.
- Style images may use different dimensions/aspects.
- Keep Krea/Qwen warm in CPU RAM where safe.
- Normal post-job cleanup frees GPU job state without blindly unloading all host model state.
- Fatal/incompatible family switches may perform a full model unload.
- Native Turbo defaults: 8 / CFG1 / Euler / simple / denoise1.
- No SageAttention or FlashAttention dependency in v1.
- Use one pinned Comfy installation; no H3-style two-Comfy retrofit.
- Heavy model layers use a Comfy-independent model root so installing Comfy afterward is safe.
- Generic D1 worker table is `image_pod_workers`, never `krea2_pod_workers`.
- Generic pod server/provisioning/lifecycle must be reusable by future image families.
- Image pod supports H3/Enhancer-class idle timeout, busy/draining state, cancellation, diagnostics, heartbeats, stale-worker recovery, provider deletion retries and callback/error handling.
- Reuse H3 pod auth master secret; no Krea-only secret.
- Preserve the existing browser image-generation polling protocol.

---

## 29. First implementation milestone

```text
35 GB container disk only
no network storage
CUDA 13 / PyTorch 2.13 cu130
Krea 2 Turbo INT8 ConvRot
Qwen3-VL-4B BF16
Qwen Image + selected Wan 2.1 VAE baked
weights stored under /opt/scenebuilder-models/krea2
one pinned Comfy installation after heavy weight layers
extra Comfy model paths configured
nodes -> workflow -> generic image runtime last
no Sage/FlashAttention
1280x720 / 720x1280
2048x1152 / 1152x2048
8 steps / CFG1 / Euler / simple / denoise1 defaults
one RunPod >=20 GB canary
warm Krea/Qwen CPU offload lifecycle proven
```

Acceptance criteria:

```text
Comfy boots and discovers the pre-baked external model root
native Krea2 + KREA2 Qwen CLIP load
BF16 encoder works
Krea/Qwen move between GPU and CPU/offload state on a 20 GB worker
job GPU allocations clear after completion
warm second job avoids unnecessary checkpoint reload
both VAEs load; Qwen is default
1K/2K aligned presets generate
seed/steps/CFG/sampler/scheduler/denoise are runtime-patchable
output uploads to R2
pod remains reusable
busy/draining/idle states behave correctly
generic image worker identity is used, not a Krea-specific control-plane table
```

After this, add the D1 image-pod lifecycle tables, LoRA cache/catalog and style-reference mode before normal frontend rollout.

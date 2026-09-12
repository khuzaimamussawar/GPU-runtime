# SceneBuilder Krea 2 Runtime Plan

Status: **architecture / implementation plan**

Scope: add Krea 2 Turbo image generation to SceneBuilder Storyboard and CharacterScreen while creating a **generic image-generation pod/control-plane layer** that can host additional image models later. Krea 2 is the first image family; it must not become the name or boundary of the reusable pod infrastructure.

This file is the source of truth for the Krea 2 runtime, generic image-pod lifecycle, D1 scheduling, LoRA catalog, thumbnail/finalization behavior, and Storyboard/Character integration.

---

## 1. Locked product decisions

### Krea 2 model/runtime

- Primary model: **Krea 2 Turbo INT8 ConvRot**.
- Diffusion checkpoint: `krea2_turbo_int8_convrot.safetensors`.
- Runtime: **native ComfyUI Krea 2 support**.
- Text encoder: **Qwen3-VL-4B BF16** (`qwen3vl_4b_bf16.safetensors`).
- Default VAE: Qwen Image VAE; Wan 2.1 VAE remains user-selectable.
- Minimum GPU: **20 GB VRAM**.
- Providers: **RunPod Pods + Novita GPU instances**.
- One pod runs **one image generation at a time**. No concurrent image generations inside one GPU pod.
- A healthy warm pod becomes reusable immediately after an image finishes.
- Pods form a **global compatible pool**, not permanent project-owned machines.
- Maximum **5 active/reserved image jobs per project** at a time.
- Maximum **5 active/reserved image pods per project** at a time.
- Capacity target: **1 pod per 3 outstanding ready/in-flight images**, capped at 5 pods per project.
- Reuse the same RunPod/Novita/R2 variables and `H3_POD_AUTH_MASTER_SECRET` used by the existing GPU control plane.

### Storage

- **No network volume.**
- **No network disk.**
- **No provider-mounted model storage.**
- Provider root/container disk: **35 GB**.
- Krea model, Qwen encoder, VAEs and the native Krea style-reference adapter are baked into Docker image layers.
- User-selectable LoRAs are not baked into Docker.
- User LoRAs may be downloaded from R2 and cached on the local 35 GB container disk.
- Generated inputs/outputs may use temporary local files during a job and are cleaned after finalization.

Provider configuration:

```text
IMAGE_POD_DISK_GB = 35
RunPod: containerDiskInGb = 35
Novita: rootfsSize = 35
Novita: networkStorages = []
RunPod: no network volume / volume mount
```

The final image must leave writable headroom for Comfy temp/output, atomic user-LoRA downloads, bounded LoRA cache and runtime scratch. Enforce a build-time rootfs size gate rather than assuming Docker layers will fit.

---

## 2. Resolution and frontend labels

The frontend should expose only the simple product labels:

```text
720p
1080p
```

The user should not need to understand Krea's internal aligned render dimensions.

### 720p

For 720p, render and deliver directly at:

```text
16:9  -> 1280 x 720
9:16  -> 720 x 1280
```

These are exact 16:9 / 9:16 and already suitable for the final output.

### 1080p

For the 1080p product tier, Krea renders at the clean aligned 2K-ish shape and the pod then downsizes to the standard delivered resolution:

```text
16:9
Krea render:  2048 x 1152
final output: 1920 x 1080

9:16
Krea render:  1152 x 2048
final output: 1080 x 1920
```

`2048x1152 -> 1920x1080` and `1152x2048 -> 1080x1920` preserve the aspect ratio exactly. Use a high-quality deterministic downscale in the pod after Comfy generation and before the final R2 upload.

Do **not** use `1368x768` as the normal product resolution. It is not a special Krea training size and it is not aligned to the preferred 16-pixel grid.

Persist both render and delivered dimensions in the durable settings snapshot:

```text
resolutionTier: 720p | 1080p
orientation: landscape | portrait
renderWidth
renderHeight
outputWidth
outputHeight
```

---

## 3. Turbo sampling/settings contract

Production defaults:

```text
steps:     8
cfg:       1.0 in the Comfy contract / effectively native no-guidance Turbo behavior
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
720p / 1080p
16:9 / 9:16
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

Initial ranges:

```text
steps:       1-50, default 8
cfg:         configurable, default 1.0
sampler:     allowlisted; Euler guaranteed
scheduler:   allowlisted; simple guaranteed
denoise:     0.0-1.0, default 1.0
seed:        explicit integer or random
VAE:         qwen_image | wan_2_1
```

At the default CFG/no-guidance path, a typed negative prompt is effectively inactive. If the user intentionally chooses a guided/higher-CFG mode, encode negative text with the same Krea text encoder and persist that mode explicitly.

Always preserve the raw prompt. If prompt enhancement is enabled, also persist the effective/enhanced prompt.

---

## 4. Model assets

Bake these assets into the image:

```text
Krea 2 Turbo INT8 ConvRot
qwen3vl_4b_bf16.safetensors
qwen_image_vae.safetensors
wan_2.1_vae.safetensors
krea2_style_reference.safetensors
```

Approximate model payload before runtime dependencies remains around the low-20-GB range plus the style adapter, so 35 GB is tight and must be enforced as a real image-size gate.

### Baked native style-reference adapter

Bake:

```text
krea2_style_reference.safetensors
```

at:

```text
/opt/scenebuilder-models/krea2/loras/krea2_style_reference.safetensors
```

Rules:

- it is a system runtime dependency, not a user LoRA;
- it is never returned by the normal LoRA catalog endpoint;
- it does not consume one of the user's 0-3 LoRA slots;
- no D1 `lora` row is required for this baked adapter;
- version/checksum may be recorded in runtime build metadata.

---

## 5. Native Comfy baseline

Use the same tested Comfy revision used by H3 unless a real Krea incompatibility requires a deliberate upgrade:

```text
COMFYUI_COMMIT = 2a68ce33b4c9ea6ee4283e618a74560cefb32694
H3 label: v0.31.0-9-g2a68ce33
comfy-kitchen = 0.2.28
```

That revision already contains native Krea 2 support, `CLIPType.KREA2`, Qwen3-VL-4B conditioning, Krea image preprocessing and INT8 ConvRot support through `comfy-kitchen`.

Do not follow moving Comfy `master` in production.

No SageAttention or FlashAttention dependency in v1. Use native Comfy/PyTorch optimized attention plus `comfy-kitchen`.

---

## 6. CUDA / PyTorch / Python baseline

Start from the same proven family as H3:

```text
CUDA: 13.0 / cu130
PyTorch: 2.13.0
Python: 3.13 target
Ubuntu: 24.04
```

Build stages may use a CUDA devel base if required. The final image should prefer the corresponding CUDA runtime base if the real ConvRot smoke test succeeds.

Strip apt, pip, Hugging Face, git and compiler/build caches from final layers.

---

## 7. Docker layer order

It is safe for Docker build layers to add Krea/Qwen/VAE files before installing Comfy. Docker build order is not runtime startup order; the final container contains all lower layers before the process starts.

The heavy assets must live outside `/opt/ComfyUI` so the later Comfy install does not collide with a pre-populated Comfy directory.

Use:

```text
/opt/scenebuilder-models/krea2/
  diffusion_models/
    krea2_turbo_int8_convrot.safetensors
  text_encoders/
    qwen3vl_4b_bf16.safetensors
  vae/
    qwen_image_vae.safetensors
    wan_2.1_vae.safetensors
  loras/
    krea2_style_reference.safetensors
```

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
   pinned requirements/comfy-kitchen
   extra model paths -> /opt/scenebuilder-models/krea2

50 image-krea2-nodes
   only required helper/custom nodes

60 image-krea2-workflow
   API workflows + manifests

70 image-krea2-runtime
   LAST layer
   generic image-pod HTTP server
   model-family adapter registry
   R2 finalization + thumbnail creation
   user-LoRA cache manager
   workflow patching
   cancellation/progress/readiness
   warm memory cleanup/offload
   global-pool idle/draining lifecycle
```

We do not need H3's historical two-Comfy/core-overlay retrofit because this layout starts cleanly from the beginning.

Changing Comfy must not invalidate Krea/Qwen/VAE layers. Changing runtime should rebuild only the final small runtime layer.

---

## 8. Hetzner layer-by-layer build

Add:

```text
.github/workflows/hetzner-krea2-build.yml
krea2/scripts/remote_build.sh
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

Mirror the H3/Enhancer temporary Hetzner builder pattern and push each successful parent target so retries reuse the heavyweight published layers.

---

## 9. Style-reference mode

Per image choose exactly one styling path in v1:

```text
A. USER LoRA mode
   0-3 user-selected D1 LoRA IDs
   no style-reference images

B. STYLE REFERENCE mode
   zero user-selected LoRAs
   1-3 style-reference images
   baked krea2_style_reference.safetensors applied automatically
```

Do not combine arbitrary user LoRAs and style-reference images in v1.

Style-reference images do not need to match one another or the output aspect ratio.

Preprocessing:

```text
1. EXIF-orient
2. convert to RGB
3. preserve each source aspect ratio
4. do not stretch references to the output shape
5. only downscale when an input exceeds the runtime safety cap
6. let pinned Krea/Qwen preprocessing handle patch-grid alignment
```

Functional tests include square + landscape + portrait references together and odd input dimensions.

---

## 10. Generic user LoRA catalog

Create a generic D1 LoRA catalog for future image/video models.

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

`display_name` is deliberately not unique. Browser/API identity is the immutable `id` only.

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

`min_steps/max_steps/recommended_steps` describe compatibility with the job's global sampling step count. They are not per-LoRA start/end schedules.

Existing Krea LoRAs keep their current R2 keys:

```text
models/lora/krea2/MinimalisticVectorArtKrea2.safetensors
models/lora/krea2/Darkchurch_style_krea2_v1.0.safetensors
```

Seed them with immutable generated IDs.

---

## 11. User-LoRA local disk + hot RAM policy

Local cache:

```text
/opt/scenebuilder-image/cache/loras/<lora_id>.safetensors
```

First use:

```text
loraId
 -> Worker validates D1 row + model support
 -> trusted R2 object key
 -> local cache hit OR R2 download to .part
 -> integrity validation
 -> atomic rename
 -> parse tensors
 -> apply model patch
```

Use immutable `lora_id`, never display name, as the cache filename.

Hot CPU cache identity should include the asset version when available:

```text
lora_id + sha256/etag/version
```

Strength is not cache identity. Reuse the same parsed tensors at a different requested strength.

If the next job uses the same LoRA set, reuse parsed CPU tensors. If it changes LoRAs, old parsed tensors may be evicted from RAM while the safetensors file remains in the bounded local disk cache.

Disk eviction removes oldest unused user LoRA files first and never touches baked model/Qwen/VAE/system-adapter assets.

---

## 12. Warm Krea/Qwen/VAE memory lifecycle

The expensive state should remain reusable in host RAM between healthy jobs.

Desired post-job state:

```text
VRAM:
  job activations/workspace/latents released
  reclaimable CUDA cache cleared

CPU RAM:
  Krea model object/weights retained where safe
  Qwen encoder retained where safe
  VAE retained where useful
  current parsed LoRA state retained under a RAM cap

Disk:
  baked checkpoints unchanged
  bounded user-LoRA cache retained
```

Normal sequence on constrained GPUs:

```text
1. activate Qwen
2. encode prompt + optional style references
3. offload Qwen to CPU
4. clear reclaimable GPU memory
5. activate Krea
6. load/apply needed user LoRAs from RAM/disk/R2
7. sample
8. activate selected VAE
9. decode
10. post-resize to final 720p/1080p dimensions
11. upload final image
12. create/upload thumbnail
13. release job GPU state
14. retain warm reusable CPU state
```

Do not perform an unconditional full model unload after every job. Full unload is reserved for incompatible family switches, fatal CUDA state, severe host-RAM pressure, or pod draining/deletion.

This follows the same concept as Enhancer's long-lived engine cache plus per-job execution cleanup.

---

## 13. Generic image pod service

Use a generic runtime identity such as:

```text
scene-builder-image-pod
```

Krea is one adapter/task family:

```text
krea2_image
```

Future image families use the same pod protocol and D1 worker tables.

Adapter registry concept:

```text
TASK_FAMILY_REGISTRY = {
  "krea2_image": Krea2Adapter,
  ...future image families...
}
```

Each adapter owns model/workflow selection, request validation, settings patching, style/reference behavior and family-specific cleanup. The HTTP server, auth, R2 finalization, thumbnails, queue state, idle timeout, callbacks, cancellation, diagnostics and provider lifecycle remain generic.

---

## 14. `image_generation_jobs` remains the durable leaf-job table

Do not create a Krea-only job table.

Every generated character/storyboard image remains one row in `image_generation_jobs`.

The current table already has useful durable fields such as:

```text
project_id
surface
target_id / target_index / target_key
dependency_target_id
model_key
aspect_ratio
image_object_key
thumbnail_object_key
thumbnail_status
cost / credits_charged
```

The image-pod path needs a migration so a job can exist **before** any RunPod/Novita instance has been assigned. The current legacy schema requires `provider` and `provider_task_id` immediately; that cannot be the source of truth for a queued generic pod job.

Recommended additions/changes:

```text
backend_kind              legacy | image_pod
task_family               krea2_image | future families
requested_provider        auto | runpod | novita
provider                  nullable actual provider
provider_task_id          nullable actual provider instance/task id
pod_worker_id             nullable image_pod_workers.id
settings_json              immutable generation snapshot
stage                      queue/runtime phase
attempt_count
retry_after
provision_retry_after
heartbeat_at
last_progress_at
error_code
error_message
```

For the image-pod path, create the row first as `queued` with no pod assignment. Provider/worker fields are filled only when scheduling actually succeeds.

Suggested statuses/phases:

```text
queued
waiting_for_capacity
waiting_for_pod
dispatching
processing
encoding_prompt
generating
decoding
resizing
uploading
finalizing
completed
failed
cancelled
```

Legacy image providers can continue using their existing path while the new fields are optional for them.

---

## 15. Generic `image_pod_workers` D1 table

Create a separate generic image-worker table. Do not reuse H3/Enhancer tables and do not call it `krea2_pod_workers`.

```sql
CREATE TABLE IF NOT EXISTS image_pod_workers (
    id TEXT PRIMARY KEY,
    service_kind TEXT NOT NULL DEFAULT 'image_generation',

    -- project_id is temporary reservation/active ownership only.
    -- A healthy idle global worker should normally have it cleared.
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

Indexes should cover reusable idle workers, temporary project reservations, provider instance lookup, heartbeat/timeout scans and delete retries.

Worker states:

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

`draining` never accepts new work.

---

## 16. Auxiliary lifecycle tables

Use D1-backed coordination rather than relying on process memory.

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

### Event replay protection

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

### Dispatch locks

Use a global scheduler lease plus short project-level admission locks/atomic conditional updates so two invocations cannot both admit the sixth active job or claim the same idle worker.

---

## 17. Frontend batching stays rolling, not a new server batch API

Keep the current Character/Storyboard behavior conceptually as it is.

The frontend maintains a rolling window of at most **10 image-generation calls**. It does not need to send one atomic 10-image GPU workload.

Example with 25 independent images:

```text
initially submit 1..10
image 3 completes -> frontend submits 11
image 7 completes -> frontend submits 12
image 1 completes -> frontend submits 13
...
```

Each `apiGenerateImage()` call owns one durable `image_generation_jobs` row and waits/polls until that image is terminal before that frontend worker takes another image.

For Storyboard, a scene that requires the previous generated scene remains dependency-gated. Do not blindly submit dependent scenes merely to fill D1.

Therefore:

```text
frontend rolling request window = up to 10
actual project GPU in-flight cap = 5
one GPU pod = one active image
```

The two limits solve different problems and should not be conflated.

No `image_pod_workloads` multi-image envelope is required in v1. Scheduling is per-image job.

---

## 18. Project admission guard: hard maximum 5 in-flight jobs

This is a hard control-plane rule and must be checked **before** posting to a pod or attempting a new provider allocation.

Define an active/reserved project job as any non-terminal image-pod job that has already consumed a scheduling slot, for example:

```text
waiting_for_pod with pod_worker_id assigned
dispatching
processing
encoding_prompt
generating
decoding
resizing
uploading
finalizing
```

Also include any equivalent legacy `submitted`/processing state if it is routed through the generic image-pod scheduler during migration.

Unassigned `queued` and `waiting_for_capacity` rows do **not** consume one of the five execution slots.

Before any scheduler path dispatches a job for project P:

```text
active = countActiveReservedImageJobs(P)
slots  = 5 - active

if slots <= 0:
    DO NOT claim another job
    DO NOT POST to a pod
    DO NOT provision another pod for that job
    DO NOT set a fake capacity error
    leave queued jobs untouched
```

If `slots > 0`, admit at most `slots` jobs for that project in that dispatcher pass.

This guard applies equally to:

```text
1-minute cron dispatcher
job-completed event-driven dispatch
worker-ready event dispatch
manual/recovery dispatch
```

This avoids the undesirable pattern where a cron blindly submits a sixth job, receives capacity/no-pod failure, then repeatedly requeues it.

---

## 19. Pod-capacity target: one pod per three outstanding images

For a project with ready/in-flight image-pod work:

```text
outstanding = readyQueuedJobs + activeReservedJobs
wantedPods  = min(5, ceil(outstanding / 3))
```

Examples:

```text
1-3 images   -> target 1 pod
4-6          -> target 2 pods
7-9          -> target 3 pods
10-12        -> target 4 pods
13+          -> target 5 pods
```

This is a provisioning target, not permission to exceed the five-job admission guard.

Always reuse compatible global idle pods first. Only provision the remaining deficit:

```text
deficit = max(0, wantedPods - compatibleReservedOrActiveWorkers)
```

A worker that is already busy is never deleted merely because `wantedPods` falls after work completes. It simply finishes, becomes global idle, and can immediately serve another project.

---

## 20. Global pool behavior

Project ownership on `image_pod_workers` is temporary while a pod is reserved or executing a project job.

After a successful image finishes:

```text
job terminal/finalized
worker current_job_id = NULL
worker status = idle
worker project_id = NULL
worker user_email = NULL
worker becomes global compatible capacity
```

Then immediately attempt to claim another compatible queued image from **any** project, subject to that target project's five-job admission limit.

Prefer family affinity where useful so a Krea-warm pod takes another Krea job before forcing an unnecessary family switch.

A pod must not wait for the 1-minute cron if a compatible queued job is already available when it becomes idle.

---

## 21. Capacity failure and requeue semantics

Separate normal queue backpressure from real terminal failures.

### A. Project already has 5 active/reserved jobs

This must be caught by the pre-dispatch admission guard. No dispatch/provision attempt occurs, so there is nothing to requeue. Queued rows remain queued without error churn.

### B. Pod dispatch race / pod reports busy before accepting this job

If another job won the worker race or the pod returns a busy/conflict response before accepting this job:

```text
reconcile worker to the real busy assignment
clear this rejected job's pod/provider assignment
put this job back to queued immediately
no user-visible failure
```

### C. New-pod provider capacity is exhausted but the project has at least one live compatible worker

This is the important Krea/image-pod behavior:

If RunPod/Novita cannot allocate the additional desired pod **and at least one compatible non-draining project worker/reservation already exists**, then the image job is still serviceable eventually by that project's existing pod capacity.

Therefore:

```text
clear failed provisional assignment
requeue job as queued
preserve it as eligible for an existing worker
record capacity telemetry for observability, not as a terminal job error
apply a provisioning cooldown before attempting another new pod
```

The job may be picked up immediately when an existing project pod becomes idle. The cooldown blocks repeated provider-create attempts; it must not block reuse of an already-running pod.

### D. Provider capacity is exhausted and the project has zero live compatible workers

Do not spin the normal queue every minute pretending capacity exists.

Move the job to:

```text
waiting_for_capacity
```

with a bounded provider-provision backoff/capacity deadline. The scheduler may retry provider allocation after the cooldown. If no compatible worker can be obtained before the configured capacity timeout/attempt budget, fail with a clear normalized `PROVIDER_CAPACITY_EXHAUSTED` error.

This keeps transient provider shortages retryable without creating an endless dispatch/requeue loop.

### E. Fatal worker/model/CUDA failure after the pod accepted the job

Use the normal retry policy only when the job is safe to retry. Drain/delete the unsafe worker. Do not classify fatal runtime corruption as simple capacity backpressure.

---

## 22. 1-minute scheduler cron

The 1-minute cron is a recovery/fill scheduler, **not a blind job submitter**.

For every project with image-pod work:

```text
1. reconcile stale worker/job assignments
2. count active/reserved image jobs
3. if active >= 5: skip dispatch for that project entirely
4. otherwise compute remaining execution slots
5. find eligible queued jobs only up to those slots
6. reuse compatible global idle workers first
7. compute 1-pod-per-3 provisioning deficit
8. provision only the real deficit and only when provider cooldown permits
9. atomically claim worker + job before POST
10. never POST a sixth active/reserved project job
```

It may also repair stale heartbeats, stale provisioning reservations and missed callbacks.

The cron must not repeatedly select a project whose five slots are already occupied merely to receive `capacity/no pod available` and requeue the same rows.

---

## 23. 15-minute provider cleanup/reconciliation cron

Use the existing 15-minute cadence, e.g.:

```text
7,22,37,52 * * * *
```

for heavier cleanup/reconciliation:

```text
delete expired/draining provider instances
retry failed provider deletions
verify billed provider instance is actually gone
clean failed provisioning workers
reconcile D1 rows against RunPod/Novita reality
clean expired deletion/dispatch locks
clean expired event nonces
```

Normal idle-worker expiry may be marked immediately/event-driven, but provider deletion must be recoverable by this cron if a callback or direct deletion attempt is lost.

---

## 24. Thumbnail generation and finalization

For generic GPU image pods, create the thumbnail **inside the pod**, not in the browser.

Legacy image providers may keep the existing frontend thumbnail path until separately migrated.

### Finalization order

```text
Comfy generates image at render dimensions
 -> pod applies final 720p/1080p output resize when required
 -> upload final image to R2
 -> generate thumbnail from the final delivered image
 -> upload thumbnail to R2
 -> emit idempotent completion callback with both object keys
 -> D1 job becomes completed
```

### Thumbnail contract

Match the current SceneBuilder thumbnail behavior:

```text
format: JPEG
max width: 700 px
quality: 0.80
preserve aspect ratio
```

Portrait thumbnails therefore keep their portrait aspect rather than being cropped into a landscape box.

Recommended generic pod implementation: Pillow high-quality resize + JPEG encode.

### D1 behavior

On success:

```text
image_object_key = <final full image key>
thumbnail_object_key = <thumbnail key>
thumbnail_status = completed
status = completed
```

If the full image succeeds but thumbnail creation/upload fails:

```text
status = completed
image_object_key = valid full image
thumbnail_status = failed
thumbnail_object_key = NULL
```

A thumbnail failure must not fail or re-charge the image generation. UI falls back to the full image and the thumbnail can be retried separately.

Completed status API should be able to return:

```json
{
  "status": "completed",
  "imageUrl": "...",
  "thumbnailUrl": "...",
  "thumbnailStatus": "completed"
}
```

The pod-generated thumbnail path is generic image infrastructure, not Krea-specific adapter code.

---

## 25. Generic image pod HTTP API

Expose:

```text
GET  /health
GET  /ready
GET  /diagnostics
GET  /diagnostics/gpu

POST /jobs
GET  /jobs/:id
POST /jobs/:id/cancel
```

One job at a time is enough for v1; no multi-image pod workload envelope is required.

Protected endpoints use:

```text
Authorization: Bearer <SCENEBUILDER_POD_TOKEN>
```

Derive the per-worker token from:

```text
HMAC(H3_POD_AUTH_MASTER_SECRET, workerId)
```

The pod should reject a second job while busy and reject every new job while draining.

---

## 26. Events, progress and error normalization

Callbacks/events:

```text
worker_ready
worker_unhealthy
job_accepted
job_started
job_progress
job_completed
job_failed
job_cancel_requested
job_cancelled
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
resizing
uploading
finalizing
completed
failed
cancelled
```

Normalize errors such as:

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
THUMBNAIL_FAILED
INVALID_LORA_ID
LORA_INCOMPATIBLE
INVALID_SETTINGS
STYLE_REFERENCE_FAILED
JOB_TIMEOUT
PROVISION_TIMEOUT
PROVIDER_CAPACITY_EXHAUSTED
PROVIDER_CREATE_FAILED
PROVIDER_DELETE_FAILED
CALLBACK_AUTH_FAILED
CANCELLED
UNKNOWN
```

Fatal CUDA/model-corruption errors drain the worker. User validation errors fail only the job and leave a healthy pod reusable.

---

## 27. Cancellation

Cancellation must be idempotent.

For active Comfy generation, use `/interrupt`.

Queued jobs can be cancelled without touching any pod. If a job is reserved but has not been accepted by the pod yet, release its worker reservation atomically and return the worker to the global idle pool when safe.

Never charge credits twice when completion/cancel callbacks race.

---

## 28. Provider routing and GPU policy

Reuse the existing Enhancer/H3 provider inventory normalization rather than maintaining unrelated GPU-name parsing.

Krea initial minimum:

```text
20 GB VRAM
```

Initial candidate tiers:

```text
20 GB
24 GB
32 GB
48 GB fallback
```

Every architecture still needs a functional Krea smoke test because ConvRot support/performance can vary.

OOM escalation:

```text
20 -> 24 -> 32 -> 48 GB
```

Do not retry the same failed GPU tier indefinitely.

Provider root/container disk remains exactly 35 GB and no network storage is attached.

---

## 29. Krea workflows

Keep explicit Krea adapter workflows:

```text
krea2/workflows/krea2_turbo.json
krea2/workflows/krea2_style_reference.json

krea2/workflows/manifests/krea2_turbo.json
krea2/workflows/manifests/krea2_style_reference.json
```

Normal path:

```text
Krea INT8 loader
Qwen3-VL-4B BF16 KREA2 CLIP
0-3 resolved model LoRA patches
positive conditioning
native zeroed negative OR explicit guided negative
latent at renderWidth/renderHeight
8/Euler/simple defaults
selected VAE
decode
pod post-resize if 1080p
R2 final image + thumbnail
```

Style-reference path:

```text
1-3 style images
baked krea2_style_reference adapter
Krea INT8
Qwen3-VL-4B BF16
selected VAE
prompt/settings
native reference conditioning
sample/decode
pod post-resize if needed
R2 final image + thumbnail
```

Do not rewrite native reference-conditioning behavior until the official graph has been reproduced successfully.

---

## 30. Provider-neutral image payload

Example:

```json
{
  "jobId": "job_123",
  "projectId": "proj_123",
  "taskFamily": "krea2_image",
  "model": "krea-2-turbo",
  "prompt": "...",
  "negativePrompt": "",
  "settings": {
    "resolutionTier": "1080p",
    "orientation": "landscape",
    "renderWidth": 2048,
    "renderHeight": 1152,
    "outputWidth": 1920,
    "outputHeight": 1080,
    "seed": 12345,
    "steps": 8,
    "cfg": 1.0,
    "sampler": "euler",
    "scheduler": "simple",
    "denoise": 1.0,
    "promptEnhance": false,
    "textEncoder": "qwen3vl_4b_bf16",
    "vae": "qwen_image",
    "styleMode": "lora",
    "loras": [
      { "loraId": "<uuid>", "strength": 0.8 }
    ]
  },
  "inputs": {
    "outputPrefix": "projects/proj_123/images/generated"
  }
}
```

Style-reference mode uses `styleMode: reference_images`, zero user LoRAs and trusted style-image object keys.

---

## 31. SceneBuilder integration

Reuse the existing browser flow:

```text
CharacterScreen / Storyboard
 -> apiGenerateImage()
 -> /api/generate-image
 -> one image_generation_jobs row
 -> generic image-pod scheduler
 -> image_pod_workers
 -> RunPod/Novita pod
 -> existing status polling/cancel semantics
 -> R2 final image + pod thumbnail
 -> existing Character/Storyboard state update
```

Do not create a Krea-only browser polling protocol.

Immutable settings snapshot must include model/quantization, task family, 720p/1080p, orientation, render/output dimensions, raw/effective prompt, negative/guidance mode, seed, steps/CFG/sampler/scheduler/denoise, VAE, text encoder profile, style mode, ordered LoRA IDs+strengths or style-image object keys, and actual provider/worker/GPU execution metadata.

Shared UI can expose:

```text
Model: Krea 2 Turbo
Resolution: 720p | 1080p
Aspect: 16:9 | 9:16
LoRA selector max 3 OR style references
Advanced generation settings
```

---

## 32. Readiness/diagnostics

`/ready` should verify:

```text
GPU visible
minimum VRAM/capability
runtime build metadata
Comfy responds
required baked model files discoverable
Krea2 model class available
KREA2 CLIP type available
comfy-kitchen backend available
baked style adapter discoverable
R2 config available
worker not draining
```

`/diagnostics` should expose safe metadata including worker status/current job, loaded family/model/VAE/text encoder, idle deadline, GPU/driver/VRAM/utilization, CUDA/PyTorch/Comfy versions, required-file checks, disk usage, LoRA cache usage, host RAM/warm state and last normalized error. Never expose secrets/tokens.

---

## 33. Tests / canaries

### Build/runtime

```text
35 GB rootfs policy respected
no network storage
CUDA 13 visible
PyTorch 2.13/cu130
single pinned Comfy discovers external baked model root
comfy-kitchen 0.2.28
native Krea2 recognized
Qwen KREA2 CLIP recognized
ConvRot works
baked style adapter discovered
no Sage/Flash dependency
```

### Resolution/final output

```text
720p 16:9: 1280x720 -> 1280x720
720p 9:16: 720x1280 -> 720x1280
1080p 16:9: 2048x1152 -> 1920x1080
1080p 9:16: 1152x2048 -> 1080x1920
```

Verify exact final dimensions, aspect ratio, post-resize quality, peak VRAM and no accidental crop.

### Thumbnail

```text
full image uploaded
700px max-width JPEG thumbnail generated in pod
thumbnail preserves aspect
D1 stores both object keys
thumbnail failure leaves image job completed
browser can close before generation completes and thumbnail still exists
```

### Style references

```text
1 / 2 / 3 images
square + landscape + portrait together
odd dimensions
zero user LoRAs enforced
baked adapter auto-applied
```

### User LoRA cache

```text
0 LoRA
both existing LoRAs individually and together
invalid ID rejected
4 user LoRAs rejected
R2 miss -> local disk cache
same LoRA next job -> hot RAM reuse
same LoRA different strength -> same parsed tensors, new patch strength
new LoRA -> old RAM tensors evictable, disk file retained
bounded LRU/watermark eviction
```

### Warm memory

```text
job-specific GPU allocations released
Krea/Qwen not unnecessarily reloaded from disk
warm CPU states survive healthy idle
same LoRA hot tensors reusable
host-memory pressure eviction works
```

### Scheduler/admission

```text
frontend may have 10 requests in flight
project scheduler never has >5 active/reserved GPU image jobs
when active count is 5, 1-minute cron does not claim/post/provision a sixth job
when one active job completes, one new slot can be filled immediately
10 ready jobs target 4 pods, not 10 pods
global idle pods are reused before provisioning
idle pod can move project A -> project B immediately
busy/draining pods reject extra work
atomic race cannot admit two jobs into the same fifth slot
```

### Capacity failure

```text
provider capacity failure + existing live compatible project pod -> job requeued, not failed
requeued job can run on that existing pod when it becomes idle
provider-create cooldown prevents every-minute allocation spam
provider capacity failure + zero live project pods -> waiting_for_capacity with bounded backoff
capacity timeout eventually becomes clear terminal PROVIDER_CAPACITY_EXHAUSTED
pod busy race requeues rejected job and reconciles true busy assignment
```

### Lifecycle

```text
provisioning -> ready -> idle
idle -> busy -> idle
idle -> global pool -> different project
idle timeout -> draining -> provider delete
lost callback recovered by cron
provider deletion retry/verification
heartbeat timeout
provision timeout
queued cancel
active interrupt cancel
callback replay protection
fatal CUDA error drains worker
normal invalid settings leave worker healthy
runtime version rollout drains old workers
```

---

## 34. Rollout order

### Phase 1 — Krea image runtime

1. Build CUDA13/PyTorch2.13 base.
2. Bake Krea INT8, VAEs, Qwen BF16 and system style adapter under `/opt/scenebuilder-models/krea2`.
3. Install one pinned Comfy layer afterward.
4. Add Krea workflows and generic final runtime.
5. Enforce 35 GB rootfs gate.
6. Prove 720p/1080p landscape+portrait paths.
7. Prove warm Krea/Qwen CPU offload.
8. Prove pod-side final resize + thumbnail upload.

### Phase 2 — generic D1 worker/control-plane lifecycle

1. Add `image_pod_workers` + indexes.
2. Add deletion locks, replay protection and dispatch/project locks.
3. Extend `image_generation_jobs` for queue-before-provider-assignment.
4. Implement RunPod + Novita provisioning with 35 GB/no network storage.
5. Implement idle/draining/reaper/error/cancellation parity with H3/Enhancer.

### Phase 3 — scheduler/global pool

1. Implement hard 5-active/reserved-jobs-per-project admission guard.
2. Implement global compatible idle pool.
3. Implement 1-pod-per-3 outstanding images, max 5.
4. Implement event-driven immediate reuse.
5. Implement 1-minute recovery/fill cron that skips projects already at five.
6. Implement provider-capacity requeue/backoff rules.
7. Implement 15-minute provider deletion/reconciliation cron.

### Phase 4 — user LoRA catalog/cache

1. Add generic `lora` + `lora_model_support`.
2. Seed existing Krea LoRAs with immutable IDs.
3. Add trusted R2 -> local disk cache.
4. Add hot parsed-LoRA RAM reuse.
5. Add bounded disk LRU/watermarks.

### Phase 5 — style references

1. Reproduce the native Krea style-reference workflow using the baked adapter.
2. Support 1-3 mixed-size references with zero user LoRAs.

### Phase 6 — SceneBuilder UI

1. Add Krea 2 Turbo model selection.
2. Add `720p | 1080p` and `16:9 | 9:16` controls.
3. Route Krea through generic image pods while preserving current rolling frontend concurrency.
4. Add D1-ID LoRA picker and style-reference alternative.
5. Preserve existing polling/cancel/final state updates.

---

## 35. Guardrails

- No network volume/network disk.
- Exactly 35 GB provider root/container disk.
- Krea/Qwen/VAEs/system style adapter baked into image.
- User LoRAs remain D1/R2 assets and may be locally cached.
- D1 `lora_id` is immutable API/cache identity; display names may collide.
- Maximum 3 user LoRAs.
- Style-reference mode uses zero user LoRAs in v1.
- Style images may have different sizes/aspects.
- One pod executes one image at a time.
- Frontend may maintain up to 10 generation calls, but project GPU admission is capped at five.
- Never attempt a sixth active/reserved project image and then rely on a capacity error to push it back.
- Global idle pods are not permanently project-owned.
- Reuse a healthy idle worker immediately; do not wait for cron.
- Provider capacity exhaustion is retryable/backpressure according to the rules above, not automatically a user job failure.
- 720p outputs are exactly 1280x720 / 720x1280.
- 1080p Krea renders are 2048x1152 / 1152x2048 and delivered as 1920x1080 / 1080x1920.
- Generic image pods create the final thumbnail inside the pod.
- Thumbnail failure does not fail the image.
- Keep Krea/Qwen warm in CPU RAM where safe.
- Normal post-job cleanup frees GPU job state without blindly unloading reusable host model state.
- Native Turbo defaults remain 8 / CFG1 / Euler / simple / denoise1.
- No SageAttention or FlashAttention dependency in v1.
- One pinned Comfy installation; no H3-style two-Comfy retrofit.
- Generic worker table is `image_pod_workers`, never Krea-specific.
- Reuse the H3 pod auth master secret; no Krea-only secret.
- Preserve the existing browser image-generation polling model.

---

## 36. First implementation milestone

```text
35 GB container disk only
no network storage
CUDA 13 / PyTorch 2.13 cu130
Krea 2 Turbo INT8 ConvRot
Qwen3-VL-4B BF16
Qwen Image + Wan 2.1 VAE baked
krea2_style_reference.safetensors baked
weights outside /opt/ComfyUI
one pinned Comfy after heavy model layers
nodes -> workflow -> generic runtime last
no Sage/Flash
720p: 1280x720 / 720x1280
1080p render: 2048x1152 / 1152x2048
1080p output: 1920x1080 / 1080x1920
8 / CFG1 / Euler / simple / denoise1 defaults
pod creates 700px JPEG thumbnail
one RunPod >=20 GB smoke test
warm Krea/Qwen CPU offload proven
```

Acceptance criteria:

```text
Comfy boots and discovers baked external model root
native Krea2 + KREA2 Qwen CLIP load
BF16 encoder works
baked style adapter is discoverable
Krea/Qwen move between GPU and CPU/offload state on a 20 GB worker
job GPU allocations clear after completion
warm second job avoids unnecessary checkpoint reload
both VAEs load
all four 720p/1080p orientations produce exact delivered dimensions
full image + thumbnail upload to R2
thumbnail failure path is non-fatal
pod remains reusable
busy/draining/idle states behave correctly
generic image-worker identity is used
scheduler never admits >5 active/reserved jobs for one project
10 queued images target 4 pods
provider capacity failure can requeue behind an existing project pod
```

# SceneBuilder Krea 2 Runtime Plan

Status: **architecture / implementation plan**

Scope: add Krea 2 Turbo image generation to SceneBuilder Storyboard and CharacterScreen using the existing GPU-runtime + SceneBuilder2 control-plane patterns proven by H3 and Enhancer.

This file is the source of truth for the Krea 2 implementation. Runtime/provider proof comes first; SceneBuilder UI and D1 integration follow after the image canaries pass.

---

## 1. Locked product decisions

### Runtime/model

- Primary model: **Krea 2 Turbo INT8 ConvRot**.
- Runtime: **native ComfyUI Krea 2 support**; do not build a parallel Diffusers service for v1.
- Text encoder: **Qwen3-VL-4B BF16**.
  - Native Comfy Krea 2 explicitly uses Qwen3-VL-4B with Krea-specific hidden-state taps.
  - Comfy-Org publishes `qwen3vl_4b_bf16.safetensors`; BF16 is therefore a supported/native encoder file, not a guessed conversion.
  - Keep FP8 as a possible later memory profile, not the v1 default.
- Diffusion checkpoint: `krea2_turbo_int8_convrot.safetensors`.
- Minimum GPU: **20 GB VRAM**.
- One generation at a time per GPU for v1.
- Providers: **RunPod Pods + Novita GPU instances**.
- Reuse H3/Enhancer auth, R2 and provider variables; do not create a third auth/config system.

### Storage

- **No network volume.**
- **No network disk.**
- **No provider-mounted model storage.**
- Runtime gets exactly **35 GB container/root filesystem disk**.
- Model/encoder/VAE files are baked into the Docker image layers.
- LoRAs are NOT baked into Docker and are NOT written to container disk.
- LoRAs are read from R2 into CPU RAM and applied in memory.
- Generated images use normal temporary runtime output only long enough to upload to project R2; do not create a persistent cache.

Provider configuration must therefore use:

```text
KREA2_POD_DISK_GB = 35
RunPod: containerDiskInGb = 35
Novita: rootfsSize = 35
Novita: networkStorages = []
RunPod: no network volume / volume mount
```

A build-size gate is mandatory because 35 GB is hard, not aspirational. Target the final unpacked root filesystem at **<=31 GB** so approximately 4 GB remains for writable runtime state, Comfy temporary output, Python caches that escape cleanup, and safety headroom.

### VAE

User can select:

```text
qwen_image   -> qwen_image_vae.safetensors
wan_2_1      -> wan_2.1_vae.safetensors
```

Rules:

- `qwen_image` is the default because it is the official/native Comfy Krea 2 VAE path.
- `wan_2_1` is the alternate path and must pass our matched-seed canary before production exposure.
- Use the compact ~254 MB Wan 2.1 VAE build, not a larger FP32 duplicate, unless the canary proves a reason to do otherwise.
- The Qwen3-VL text encoder is independent from VAE choice. The same BF16 Qwen3-VL-4B conditioning can feed a job decoded with either supported VAE; VAE selection is a latent decode choice, not a text-encoder mode.

### LoRAs

- **0 to 3 LoRAs per generation.**
- Existing R2 files stay exactly where they are:

```text
models/lora/krea2/MinimalisticVectorArtKrea2.safetensors
models/lora/krea2/Darkchurch_style_krea2_v1.0.safetensors
```

- Existing thumbnails are UI-only assets:

```text
models/lora/krea2/minimalist_vector_art_thumbnail.png
models/lora/krea2/dark_church_style_thumbnail.png
```

- GPU runtime never downloads thumbnails.
- GPU runtime never selects a LoRA by display name, slug or filename.
- Every LoRA gets an immutable **D1 `lora_id`** (UUID/text primary key).
- Browser/API requests use `loraId` only because two LoRAs may legitimately have the same visible name.
- Display names are explicitly **not unique**.
- R2 object key identifies the physical asset after the Worker resolves a trusted D1 row.
- Optional SHA-256 metadata may exist for integrity/debugging, but it is not the ID, filename or R2 naming scheme and is not required for the two existing objects.

---

## 2. Native ComfyUI Krea 2 facts we rely on

The pinned ComfyUI revision already used by H3 contains native Krea 2 support:

```text
COMFYUI_COMMIT = 2a68ce33b4c9ea6ee4283e618a74560cefb32694
Comfy revision label used by H3: v0.31.0-9-g2a68ce33
comfy-kitchen = 0.2.28
```

That exact revision already contains:

- native `Krea2` model detection/model class;
- native `CLIPType.KREA2`;
- native Krea 2 Qwen3-VL-4B text encoder implementation;
- INT8/ConvRot quantization support through `comfy-kitchen`;
- CUDA backend path that expects cu130 or newer for the optimized quantized operations.

Therefore v1 should use **the exact same pinned Comfy source artifact as H3**, not current moving `master`, unless a Krea-specific canary proves that we need to advance the shared pin.

Do not install an older Comfy source from the base layer and call it final. Apply the immutable H3-style Comfy overlay **after the heavy Krea/Qwen/VAE layers**, as described below.

---

## 3. CUDA / PyTorch / Python baseline

Use the same proven software family as H3:

```text
CUDA: 13.0 / cu130
PyTorch: 2.13.0
Python: 3.13 target
Ubuntu: 24.04
```

This is not just consistency: the pinned Comfy quantization code explicitly enables the optimized CUDA `comfy-kitchen` path on CUDA 13+ and warns that cu130 or newer is required for those optimized operations.

### Final-image size rule

The H3 base currently starts from a CUDA `devel` image. Krea has a hard 35 GB container disk and carries roughly:

```text
Krea 2 Turbo INT8 ConvRot  ~13.5 GB
Qwen3-VL-4B BF16           ~8.9 GB
Qwen Image VAE             ~0.25 GB
Wan 2.1 VAE                ~0.25 GB
-----------------------------------
model payload               ~22.9 GB
```

So the Krea build should use a **multi-stage strategy**:

- compilation/build stages may use CUDA 13 `devel` when required;
- the final runtime should prefer the matching CUDA 13 cuDNN **runtime** base if the Krea/Comfy/comfy-kitchen canary passes;
- PyTorch remains 2.13/cu130;
- never trade correctness for size: if the slim runtime base fails a real ConvRot canary, retain the required runtime libraries and reduce size elsewhere;
- remove apt lists, pip cache, git history, HF cache and builder artifacts from final layers.

The 35 GB requirement never changes into a network-volume workaround.

---

## 4. Docker layering: Comfy comes after the heavy Krea lineage

Use the same successful pattern as H3 `pod-models`: build an immutable Comfy artifact separately, then apply it after the heavyweight model lineage. This prevents a Comfy source update from forcing all model files to rebuild/redownload.

Recommended graph:

```text
00 krea2-base
   CUDA 13 + Python 3.13 + PyTorch 2.13/cu130 + system/runtime packages
   NO final Comfy source dependency

10 krea2-comfyui-artifact          (parallel small artifact)
   fetch exact H3 COMFYUI_COMMIT
   archive tracked Comfy source overlay
   preserve/pin matching requirements incl. comfy-kitchen 0.2.28

20 krea2-model-int8
   FROM krea2-base
   add krea2_turbo_int8_convrot.safetensors

30 krea2-vaes
   FROM krea2-model-int8
   add qwen_image_vae.safetensors
   add wan_2.1_vae.safetensors

40 krea2-qwen-bf16
   FROM krea2-vaes
   add qwen3vl_4b_bf16.safetensors

50 krea2-core
   FROM krea2-qwen-bf16
   COPY/APPLY krea2-comfyui-artifact overlay here
   install the pinned Comfy requirements
   verify native Krea2 + ConvRot + KREA2 CLIP type

60 krea2-nodes
   only SceneBuilder/Krea helper nodes actually required
   includes in-memory R2 LoRA helper if implemented as a node

70 krea2-runtime
   job contract, R2 client, workflow patcher, output upload,
   progress, cancellation, RAM cleanup

80 krea2-workflow
   baked API workflow + manifest only

90 krea2-pod
   HTTP pod server, readiness, lifecycle
```

Consequences:

- Comfy update rebuilds `core` and later layers but **not** Krea/Qwen/VAE weight layers.
- Node change rebuilds only `nodes` and later layers.
- Runtime change rebuilds only `runtime` and later layers.
- Workflow change rebuilds only `workflow` + `pod`.
- LoRA addition/change rebuilds **nothing**.

Final image:

```text
khuxaima/scenebuilder-krea2-pod:latest
```

Add a final CI/build check that starts the final image, records `du`/filesystem use, and fails if the rootfs leaves insufficient headroom inside the 35 GB provider disk.

---

## 5. Hetzner layer-by-layer build

Add a dedicated workflow:

```text
.github/workflows/hetzner-krea2-build.yml
krea2/scripts/remote_build.sh
```

Mirror H3/Enhancer temporary builder behavior:

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
comfyui-artifact
model-int8
vaes
qwen-bf16
core
nodes
runtime
workflow
pod
```

Push every successful parent to Docker Hub so later attempts reuse published parents instead of repeating heavyweight downloads.

---

## 6. Resolution contract

SceneBuilder should use exact project-friendly 16:9 / 9:16 presets rather than expose Comfy's raw megapixel calculator to normal users.

Lock these product presets initially:

```text
1K landscape: 1280 x 720
1K portrait:   720 x 1280

2K landscape: 1920 x 1080
2K portrait:  1080 x 1920
```

Why:

- exact 16:9 / 9:16, matching Storyboard/project framing;
- 1280x720 is the practical SceneBuilder "1K class" widescreen preset;
- 1920x1080 is ~2.07 MP and stays under Krea's 2048-per-side practical 2K envelope;
- both are multiples of 8 and avoid a post-generation crop just to restore project ratio.

For reference only, native Comfy's `ResolutionSelector` defines 1.0 MP as `1024*1024` total pixels. At 16:9 that calculator lands around 1368x768, and 2.0 MP around 1928x1088. SceneBuilder deliberately uses the cleaner exact-video presets above instead.

Worker validates only approved presets for normal UI. Advanced/admin support for additional Krea-native aspect ratios can be added later without changing the runtime contract.

---

## 7. Sampling/settings contract: user controls the generation

Native Turbo defaults remain:

```text
steps:     8
cfg:       1.0
sampler:   euler
scheduler: simple
denoise:   1.0
seed:      random unless pinned
VAE:       qwen_image
LoRAs:     none
```

These are defaults, **not hard-coded hidden constants**. The user can control:

```text
prompt
prompt enhancement on/off
negative prompt
resolution tier / portrait-landscape preset
seed / random seed
steps
CFG
sampler
scheduler
denoise
VAE
0-3 LoRAs
strength for every selected LoRA
```

### UI/API ranges

Initial v1 advanced controls:

```text
steps:       1-50, default 8
cfg:         configurable, default 1.0
sampler:     dropdown from our canary-approved Comfy sampler allowlist; Euler guaranteed
scheduler:   dropdown from our canary-approved scheduler allowlist; simple guaranteed
denoise:     0.0-1.0, default 1.0
seed:        explicit integer or random
VAE:         qwen_image | wan_2_1 after Wan canary passes
```

Do not silently accept arbitrary sampler/scheduler strings. The user controls them through an allowlisted dropdown so saved jobs stay reproducible on our pinned image.

### Negative prompt

The official/native Turbo graph uses:

```text
positive conditioning
  -> ConditioningZeroOut
  -> KSampler negative
```

At default `CFG = 1.0`, negative guidance is effectively inactive; this is the native Turbo/no-guidance behavior.

SceneBuilder can still expose a negative-prompt field as an advanced control:

- at CFG 1, show it as **inactive / no effect**;
- when the user raises CFG above the native default, the runtime may encode the supplied negative text with the same Krea text encoder and feed it to KSampler;
- this guided/negative mode must be included in the Krea canary matrix because Turbo is distilled for the low/no-guidance path and higher CFG can change quality.

Do not pretend negative prompting is a native-default Turbo feature when CFG is 1.

### Prompt control

Always send the raw user prompt in the durable job snapshot.

If we enable native Krea/Comfy prompt enhancement, expose it explicitly:

```text
promptEnhance: true | false
```

Do not overwrite the user's stored raw prompt with an enhanced string. Store both raw prompt and effective/enhanced prompt in execution metadata when enhancement is used so regeneration is reproducible.

---

## 8. LoRA IDs, compatibility and steps

Browser payload uses immutable D1 IDs:

```json
"loras": [
  { "loraId": "7c5d...uuid...", "strength": 0.8 },
  { "loraId": "1a03...uuid...", "strength": 0.65 }
]
```

Never use this as identity:

```text
display name
slug
filename
thumbnail name
R2 key supplied by browser
```

### Important: "LoRA steps"

For v1, LoRAs use native/model patching at a constant strength for the whole generation. There is **no invented per-LoRA start-step/end-step scheduler**.

D1 stores:

```text
min_steps
max_steps
recommended_steps
```

These describe which **global generation `steps` value** the LoRA is validated for. The user controls the global steps field. The LoRA picker can show the recommended value/range and the Worker validates the combination.

If we later want per-LoRA start/end sampling schedules, that is a separate feature requiring a tested scheduling patch/node and should not be confused with compatibility metadata.

---

## 9. Generic D1 LoRA catalog

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

    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
```

Notes:

- `id` is an immutable generated UUID/text ID and is what UI/API jobs persist.
- `display_name` is **not unique**.
- Two or twenty LoRAs may share the same visible name.
- `r2_object_key` is unique because it identifies the physical stored asset.
- `thumbnail_object_key` is UI-only.
- `sha256` is optional metadata, not identity.
- `lora_type`: `image | video | both`.

### `lora_model_support`

Normalize quantization into the compatibility key instead of hiding it in a JSON list:

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

Examples for `quantization`:

```text
int8_convrot
bf16
fp8_scaled
*
```

Examples for `apply_target`:

```text
model_only
clip_only
model_and_clip
```

This allows the same physical R2 file to support multiple future models/quantizations without duplicating it.

Seed the two existing Krea files with two newly generated immutable `lora.id` values. Do not infer the IDs from their display names.

---

## 10. LoRA runtime: R2 -> CPU RAM -> Comfy patch

Hard requirement: **no LoRA safetensors file on container disk**.

Flow:

```text
browser loraId
  -> SceneBuilder D1 lora row
  -> D1 lora_model_support row
  -> validate model/quantization/steps/strength/enabled
  -> Worker sends trusted descriptor with exact R2 object key
  -> Pod R2 GetObject
  -> bytes in host RAM
  -> safetensors.torch.load(bytes)
  -> CPU tensors/state dict
  -> comfy.sd.load_lora_for_models(...)
  -> patched MODEL (and CLIP only if catalog explicitly requires it)
```

For normal Krea style LoRAs, default:

```text
model strength = requested strength
clip strength = 0
apply_target = model_only
```

Apply at most three in deterministic request order.

After patch creation:

- release raw object byte buffers;
- release unneeded temporary dictionaries;
- allow Comfy's model/offload machinery to retain only the CPU patch tensors needed by the current model;
- clear job-owned LoRA state on cancellation/failure;
- optional warm in-process RAM cache is allowed only with a strict RAM cap/LRU; it is never correctness-critical.

If a future Comfy API absolutely requires a path, `/dev/shm`/tmpfs is the only fallback. Normal implementation is direct memory loading.

---

## 11. Native workflow / manifest

Files:

```text
krea2/workflows/krea2_turbo.json
krea2/workflows/manifests/krea2_turbo.json
```

Baseline graph:

```text
UNETLoader
  krea2_turbo_int8_convrot.safetensors

CLIPLoader
  qwen3vl_4b_bf16.safetensors
  type = krea2

SceneBuilder in-memory MODEL LoRA patch chain (0-3)

CLIPTextEncode positive
optional CLIPTextEncode negative when guided advanced mode is active
otherwise ConditioningZeroOut for native Turbo default

EmptyLatentImage
KSampler
VAELoader / selected VAE
VAEDecode
SaveImage / runtime output
```

Manifest patch points:

```text
prompt
negativePrompt / negative mode
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

LoRA descriptors are resolved before workflow execution and never expose arbitrary filenames to the browser.

---

## 12. Provider-neutral payload

Example:

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
    "resolutionTier": "1k",
    "seed": 12345,
    "randomSeed": false,
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

The Worker writes the immutable request snapshot into the existing durable image job metadata before dispatch.

---

## 13. Attention implementation: no SageAttention / FlashAttention dependency in v1

Do **not** add SageAttention, FlashAttention or a special attention custom node to the Krea v1 image.

Native Comfy Krea 2 calls Comfy's core `optimized_attention_masked` path. With PyTorch 2.13/cu130 and the pinned Comfy stack, use the normal Comfy/PyTorch optimized attention selection plus `comfy-kitchen` for INT8 ConvRot.

Reasons:

- no Krea-specific Sage/Flash requirement exists in the native model path;
- fewer compiled packages means less image size and fewer GPU-architecture compatibility problems;
- 35 GB disk budget benefits from avoiding unnecessary compiled stacks;
- we already need per-GPU ConvRot qualification; do not add another optimization variable before baseline measurements.

Only add an alternate attention backend later if a controlled benchmark proves a meaningful benefit on the exact RunPod/Novita GPU classes and it does not break portability.

---

## 14. GPU policy

Reuse the Enhancer GPU catalog and nominal-VRAM tolerance.

```text
KREA2_MIN_VRAM_GB = 20
```

Eligible RunPod classes begin with existing >=20 GB catalog entries such as RTX 4000 Ada, RTX A4500, L4, 4090, A5000, 3090, RTX PRO 4000, 5090, RTX PRO 4500 variants, L40/L40S, RTX 6000 Ada, A40 and A6000.

Current Novita candidates are 4090, 5090, RTX 6000 Ada and L40S.

Every GPU class still requires a Krea canary because INT8 ConvRot speed/behavior can differ by architecture.

OOM escalation:

```text
20 GB -> 24 GB -> 32 GB -> 48 GB
```

Never repeat the same failed VRAM tier indefinitely.

---

## 15. VRAM / host-RAM sequence

BF16 Qwen3-VL-4B is intentionally larger than the FP8 encoder, so constrained GPUs must not keep everything resident unnecessarily.

Sequence:

```text
1. prepare Qwen3-VL-4B BF16
2. encode positive prompt (and negative only if guided mode is active)
3. finish optional prompt-enhancement work
4. offload/unload text encoder from VRAM
5. free reclaimable CUDA cache
6. load/activate Krea INT8 model
7. fetch/apply requested LoRAs from R2 into CPU RAM
8. sample
9. load/activate selected VAE as required
10. decode
11. upload output
12. clear job-specific RAM/VRAM state while leaving reusable base model state where safe
```

Initial host-RAM target remains at least 32 GB; prefer larger host RAM where provider inventory makes it cheap, especially because LoRAs are intentionally RAM-only.

---

## 16. RunPod / Novita lifecycle

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
encoding_prompt
generating
decoding
uploading
completed
failed
cancelled
```

One active job per GPU for v1. Warm workers can accept later Krea jobs until idle expiry.

---

## 17. SceneBuilder2 integration

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

Extend `image_generation_jobs` only with generic GPU-routing/snapshot fields if missing, such as:

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

Persist in `settings_json`:

```text
model key + quantization
resolution tier + width/height
raw prompt
promptEnhance flag
effective/enhanced prompt when used
negative prompt
seed/random-seed
steps/cfg/sampler/scheduler/denoise
VAE
text encoder profile
ordered loraId + strength list
requested compute backend
```

Shared UI component:

```text
<LoraSelector modelKey="krea-2-turbo" maxSelected={3} />
```

Catalog results return `id` plus display metadata. The UI renders names/thumbnails but persists IDs.

---

## 18. D1/catalog API behavior

Conceptual query:

```text
GET /api/loras?model=krea-2-turbo&quantization=int8_convrot&type=image
```

Return:

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

Never return runtime credentials. Never let browser-provided `displayName` or R2 key control which safetensors object is loaded.

Worker resolves `loraId` again at dispatch time so a stale/forged browser payload cannot bypass compatibility rules.

---

## 19. Tests / canaries

### Build/runtime

```text
final rootfs fits 35 GB disk with >=4 GB target headroom
no network volume configured
CUDA 13 visible
PyTorch 2.13 + cu130
same pinned H3 Comfy commit applied after heavy layers
comfy-kitchen 0.2.28 present
native Krea2 model recognized
Qwen3-VL-4B KREA2 CLIP recognized
ConvRot backend works
no SageAttention/FlashAttention dependency
```

### Resolution matrix

Matched prompt/seed/settings:

```text
1280x720
720x1280
1920x1080
1080x1920
```

Record VRAM, RAM, generation time and decode time.

### VAE matrix

Same seed/prompt/resolution:

```text
Qwen Image VAE
Wan 2.1 VAE
```

Compare correctness, color, detail, artifacts, decode time and memory. Wan is not exposed to normal users until this passes.

### Settings matrix

At minimum verify:

```text
8 / CFG1 / Euler / simple / denoise1 native default
custom seed
random seed
custom steps
custom CFG
approved alternate sampler/scheduler entries
negative prompt inactive at CFG1
guided negative-prompt mode when CFG is raised
prompt enhancement on/off if enabled
```

### LoRA matrix

```text
0 LoRA
MinimalisticVectorArt only
Darkchurch only
both existing LoRAs together
3-LoRA test when a third compatible asset exists
same display name on two different D1 IDs -> correct asset selected by ID
invalid ID rejected
4 LoRAs rejected
step incompatibility rejected/warned according to catalog policy
strength outside allowed range rejected
no LoRA safetensors written to container disk
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
RAM cleanup after cancelled/failed LoRA job
```

---

## 20. Rollout order

### Phase 1 - image/runtime proof

1. Build slim CUDA 13/PyTorch 2.13 Krea base.
2. Build the parallel pinned Comfy artifact using the exact H3 Comfy commit.
3. Build Krea INT8 -> both VAEs -> BF16 Qwen heavy chain.
4. Apply the pinned Comfy overlay after those heavy layers.
5. Verify final image fits the 35 GB disk policy.
6. Reproduce native Turbo 8 / CFG1 / Euler / simple / denoise1.
7. Test all four SceneBuilder resolution presets.
8. Pass a 20 GB worker canary with Qwen offload.

### Phase 2 - RAM-only LoRAs

1. Create generic `lora` + `lora_model_support` schema.
2. Seed the two existing R2 LoRAs with immutable UUID IDs.
3. Implement R2 -> RAM -> `load_lora_for_models`.
4. Test 0-3 LoRAs and name collisions.
5. Prove no LoRA file appears on container disk.

### Phase 3 - provider pods

1. Add Krea pod server/lifecycle.
2. Hard-code provider disk request to 35 GB.
3. Explicitly configure no network volume/storage.
4. Reuse H3 auth and provider/R2 vars.
5. Add RunPod/Novita allocation, readiness, warm reuse, OOM escalation and deletion.

### Phase 4 - SceneBuilder control plane

1. Add `krea-2-turbo` provider path to existing image jobs.
2. Preserve current async status/cancel/finalization APIs.
3. Add full immutable settings snapshot.
4. Add LoRA catalog endpoint by model/quantization/type.

### Phase 5 - UI

1. Add Krea 2 Turbo to CharacterScreen and Storyboard.
2. Add 1K/2K portrait/landscape presets.
3. Add shared max-3 LoRA selector using IDs.
4. Add Advanced controls for seed, steps, CFG, sampler, scheduler, denoise, negative prompt, VAE and prompt enhancement.
5. Keep native defaults easy while allowing the user to change them.

---

## 21. Guardrails

- No network volume/network disk.
- Container/root disk is 35 GB.
- No LoRA persistent-disk cache.
- Never identify LoRAs by display name.
- D1 immutable `lora_id` is the API/job identity.
- Never accept an arbitrary browser R2 object key/path.
- Maximum 3 LoRAs.
- User controls generation settings, but sampler/scheduler values come from the tested allowlist.
- Native Turbo defaults remain 8 / CFG1 / Euler / simple / denoise1.
- Negative prompt is clearly inactive at CFG1 native mode.
- No SageAttention or FlashAttention dependency in v1.
- Use the same pinned Comfy commit as H3 unless a Krea canary forces a deliberate upgrade.
- Apply Comfy after Krea/Qwen/VAE heavyweight layers so Comfy changes do not invalidate them.
- No separate browser polling system for Krea.
- No separate Krea auth master secret.

---

## 22. First implementation milestone

```text
35 GB container disk only
no network storage
CUDA 13 / PyTorch 2.13 cu130
Krea 2 Turbo INT8 ConvRot
Qwen3-VL-4B BF16
Qwen Image VAE + Wan 2.1 VAE baked
same pinned H3 Comfy commit applied after heavy model layers
no Sage/FlashAttention
1280x720 / 720x1280
1920x1080 / 1080x1920
8 steps / CFG1 / Euler / simple / denoise1 defaults
one RunPod >=20 GB canary
no LoRA in milestone 1
```

Acceptance criteria:

```text
final image/rootfs passes 35 GB size gate
Comfy boots
native Krea2 + Qwen KREA2 CLIP load
BF16 encoder conditions successfully
encoder offloads before constrained-GPU diffusion
both VAE files load; Qwen is default
all four resolution presets generate
seed/steps/CFG/sampler/scheduler/denoise are runtime-patchable
output uploads to R2
pod remains reusable for a second job
```

Then add D1-ID-based RAM-only LoRAs before frontend rollout.
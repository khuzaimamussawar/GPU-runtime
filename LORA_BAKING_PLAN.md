# LoRA Baking Plan

Status: design only. This document records the current GPU-runtime layout and
the required implementation sequence. It does not make a LoRA available to any
runtime by itself.

## Goal

Let a verified SceneBuilder LoRA that already lives in R2 be baked into one or
more compatible GPU runtime images. A bake must produce complete deployable
runtime images, not merely a parent layer. The result must carry the original
LoRA UUID, R2 object key, SHA-256, byte size, compatible model family, image
tag, and exact in-container path.

The stable runtime path for H3 LoRAs is:

```text
/opt/ComfyUI/models/loras/<file-name>.safetensors
```

The stable Krea model-root path is:

```text
/opt/scenebuilder-models/krea2/loras/<file-name>.safetensors
```

## Current layout: exact Docker chains

### H3 serverless

The serverless workflow is **before** the LoRA layer. This is the current
source relationship, not a proposed one:

```text
base -> qwen-nvfp4 -> qwen-all -> fl2va-base ------------------+
base -> comfyui -------------------------------------------------+-> fl2va-workflow
base -> sageattention -------------------------------------------+
base -> custom-nodes --------------------------------------------+
                                                                  -> fl2va-loras
                                                                  -> runpod-fl2va
                                                                  -> novita-fl2va

base -> qwen-nvfp4 -> qwen-all -> ref2va-base ------------------+
base -> comfyui --------------------------------------------------+-> ref2va-workflow
base -> sageattention --------------------------------------------+
base -> custom-nodes ---------------------------------------------+
                                                                   -> ref2va-loras
                                                                   -> runpod-ref2va
                                                                   -> novita-ref2va
```

`Dockerfile.fl2va-loras` inherits `minimax-h3-fl2va-workflow`; `Dockerfile.ref2va-loras`
inherits `minimax-h3-ref2va-workflow`. Both presently only create
`/opt/ComfyUI/models/loras`; neither copies an adapter yet.

### H3 combined pod

The combined pod has a different ordering. Its LoRA layer is **before** the
combined workflow layer:

```text
base -> qwen-nvfp4 -> qwen-all -> fl2va-base --------------------+
base -> qwen-nvfp4 -> qwen-all -> ref2va-base --------------------+-> pod-models
base -> comfyui ----------------------------------------------------+
base -> sageattention ----------------------------------------------+
base -> custom-nodes -> pod-nodes -> pod-loras -> pod-workflow -> pod

pod-models ---------------------------------------------------------> pod-nodes
```

`Dockerfile.pod-workflow` inherits `minimax-h3-pod-loras`; `Dockerfile.pod`
then inherits `minimax-h3-pod-workflow`.

### Krea 2

Krea has no baked user-LoRA layer today:

```text
base -> model-int8 -> vaes -> qwen-bf16 -> system-adapters
     -> comfyui -> nodes -> workflow -> runtime
```

`system-adapters` bakes only Krea's required `krea2_style_reference.safetensors`
at `/opt/scenebuilder-models/krea2/loras/` and verifies its SHA-256. User LoRAs
are currently trusted D1/R2 records downloaded at job time into
`/opt/scenebuilder-image/cache/loras/`.

## Current blockers

1. H3 contains no LoRA loader node in `workflows/fl2va_master.json` or
   `workflows/ref2va_master.json`, and no H3 request/runtime code resolves a
   selected LoRA. Copying files into an H3 image therefore cannot affect output.
2. Existing `fl2va-loras`, `ref2va-loras`, and `pod-loras` are empty placeholder
   layers. They have no manifest, no download/copy step, and no integrity check.
3. Krea graph patching and `image_pod.media.materialize_user_loras` assume an
   R2 object key and local cache. Baked user LoRAs need an explicit trusted
   resolution path; they must not pretend to be cached R2 downloads.
4. The current GitHub Actions build forms accept only target names and tags.
   They accept no LoRA UUID, R2 key, SHA, byte size, or compatibility selection.
5. Existing validation checks pinned Comfy/SageAttention/KJNodes/Spectrum and
   validates workflow JSON. It does not prove baked LoRA file integrity or that
   a selected LoRA is actually connected to an H3/Krea graph.

## One bake workflow, not chained workflow dispatches

Add one manually-dispatched `Bake LoRA` workflow in this repository. It creates
one temporary Hetzner builder and runs all required targets in order. It must
not dispatch the existing Hetzner workflows one by one: that would create
multiple builders, allow mixed parent tags, and leave partial final images.

Inputs:

```text
lora_id              immutable D1 UUID
r2_object_key        must begin models/lora/
expected_sha256      lowercase 64-character SHA-256
expected_size_bytes  positive integer
file_name            safe .safetensors filename
compatibility        explicit selected runtime families
image_tag            immutable release tag, never only latest
```

The workflow must reject arbitrary URLs. It downloads only the trusted R2
object using R2 credentials, streams it to the temporary builder, verifies byte
size and SHA-256 before Docker receives it, and fails closed on mismatch.

The build context must contain a generated, temporary bake manifest such as:

```json
{
  "loraId": "uuid",
  "r2ObjectKey": "models/lora/example.safetensors",
  "fileName": "example.safetensors",
  "sha256": "...",
  "fileSizeBytes": 123,
  "targets": ["h3_fl2va", "h3_ref2va", "h3_pod"]
}
```

Do not commit the downloaded safetensors file or its generated bake manifest to
Git. The builder uses them only for that release tag.

## Parent-tag versus output-tag rule

The current H3 and Krea remote build scripts have a limitation: `IMAGE_TAG` is
used both for `FROM` parent images and for the image that is pushed. A new tag
such as `lora-abc123` would therefore fail if it tries to pull
`fl2va-workflow:lora-abc123` before that parent exists. Rebuilding every parent
layer only to create a new LoRA tag would be wasteful and is not the design.

The bake implementation must add two distinct values:

```text
base_image_tag    existing, complete, pinned parent release
output_image_tag  new immutable bake release
```

For example, to bake an FL2VA LoRA while refreshing the custom-node artifact:

```text
build custom-nodes
  FROM base:<base_image_tag>
  PUSH minimax-h3-custom-nodes:<output_image_tag>

build fl2va-workflow
  FROM fl2va-base:<base_image_tag>
  COPY comfyui and sageattention from <base_image_tag>
  COPY custom-nodes from <output_image_tag>
  PUSH minimax-h3-fl2va-workflow:<output_image_tag>

build fl2va-loras
  FROM fl2va-workflow:<output_image_tag>
  PUSH minimax-h3-fl2va-loras:<output_image_tag>

build runpod-fl2va
  FROM fl2va-loras:<output_image_tag>
  PUSH minimax-h3-runpod-fl2va:<output_image_tag>
```

The same principle applies to every branch below. The bake workflow records the
resolved parent image digests and the pushed output image digests in its release
manifest. It must never rely on a later-mutated `latest` tag for reproducibility.

## Required target batches

The exact rebuild batch depends on compatibility.

### FL2VA only

```text
custom-nodes
fl2va-workflow
fl2va-loras
runpod-fl2va and novita-fl2va (parallel leaves)
```

### Ref2VA only

```text
custom-nodes
ref2va-workflow
ref2va-loras
runpod-ref2va and novita-ref2va (parallel leaves)
```

### Both serverless H3 families and the combined pod

```text
custom-nodes                             build once from the pinned base tag
fl2va-workflow -> fl2va-loras -> [runpod-fl2va, novita-fl2va]
ref2va-workflow -> ref2va-loras -> [runpod-ref2va, novita-ref2va]
pod-nodes -> pod-loras -> pod-workflow -> pod
```

The same verified file can be copied into more than one target layer. Pod and
serverless do not share one physical Docker image layer, even though the final
in-container filename/path is identical.

### Krea 2 only

Add a `krea2-user-loras` target with this chain:

```text
qwen-bf16:<base-tag>
  -> system-adapters:<bake-tag>
  -> comfyui:<bake-tag>
  -> nodes:<bake-tag>
  -> workflow:<bake-tag>
  -> user-loras:<bake-tag>
  -> runtime:<bake-tag>
```

This keeps user assets out of Krea's base/model/Qwen/system-adapter layers and
rebuilds the final runtime after the new user-LoRA layer is present.

## Runtime implementation order

### 1. H3 graph and request contract

Before enabling H3 baking, add an ordered list of at most four LoRAs to both H3
request contracts, resolve only catalog-approved UUIDs, and patch validated H3
LoRA loader nodes into the FL2VA/Ref2VA workflow graphs. The H3 runtime must verify:

- selected LoRA model compatibility;
- baked file exists at the expected path;
- selected filename matches the catalog record;
- requested strength is within catalog bounds;
- the resulting graph's ordered loader chain is correct.

No H3 LoRA appears in SceneBuilder's selectable list until this is proven by an
end-to-end canary.

Do not assume that Krea's standard `LoraLoaderModelOnly` graph patch is valid
for MiniMax H3. Krea's current LoRA path is implemented and tested; H3 has no
LoRA path at all. The one-time H3 rollout must first smoke-test an
H3-compatible LoRA against the INT8 ConvRot model and prove the correct loader
and graph connection. Only then may the runtime dynamically insert that proven
loader. Implement four explicitly ordered loader slots, each with individual
catalog strength limits, and require real H3 canaries for one, two, three, and
four compatible LoRAs. No custom-node rebuild is required for the loader itself
if core-node support is proven, but the bake release still rebuilds
`custom-nodes` so it picks up any deliberately updated pinned custom-node code.

### 2. Stable generic bake layer implementation

Convert each current H3 placeholder LoRA Dockerfile into a generic manifest
consumer. It must copy only the verified temporary input and run `sha256sum -c`
against the generated manifest. It must not contain a hardcoded named LoRA.

For Krea, add `Dockerfile.krea2-user-loras`, inheriting Krea's workflow image,
and change `Dockerfile.krea2-runtime` to inherit it. Krea's catalog resolver
must distinguish `storage_source=r2` from `storage_source=baked`; the latter
resolves the immutable file in the baked model-root path and never calls R2.

### 3. Bake manifest output

After every final image push, the workflow writes a machine-readable release
manifest. It records only targets actually rebuilt and their immutable image
references/digests. A failed target produces no publishable baked state.

### 4. SceneBuilder/D1 publication

Only after the bake workflow fully succeeds may the UI/API mark that LoRA as
`baked` for a particular compatible runtime. The record needs per-runtime
entries, because one LoRA can be baked for FL2VA but not Ref2VA, or vice versa.

The UI's existing allowed-email gate governs people using the LoRA management
screen. It does not authenticate GitHub Actions. If automatic D1 publication is
desired, use a narrowly scoped build callback credential; otherwise the allowed
user imports the signed release manifest after the successful Action run.

## Pinned runtime rules

The bake workflow must preserve the existing pinned dependency graph:

- ComfyUI: `2a68ce33b4c9ea6ee4283e618a74560cefb32694`
- SageAttention 2.2.0: `eb615cf6cf4d221338033340ee2de1c37fbdba4a`
- KJNodes: `073efb07419f56cc714e099a82e49fbc23ad9263`
- Spectrum MiniMax H3 0.2.1: `1427dfe96029985be9cbc0033abb44bf2e7adc4f`
- Krea Comfy Kitchen: `0.2.28`

It must build with a new immutable release tag, for example
`lora-<short-sha>-<utc-build-id>`, rather than changing only `latest`. Provider
deployments and the D1 manifest should refer to the immutable result. `latest`
can be moved only after a successful canary.

Every bake release intentionally rebuilds the layers from `custom-nodes` onward
for H3, and from `system-adapters` onward for Krea. This means the release can
include deliberate custom-node/workflow/Comfy changes while preserving the
heavy H3 diffusion, Krea Turbo, Qwen, and VAE model layers from the selected
`base_image_tag`. The distinct parent/output tag inputs are required for this
to work without rebuilding those heavy ancestors.

## Acceptance tests

1. invalid R2 prefix, file extension, size, SHA, or duplicate filename fails
   before an image is built;
2. only compatible target batches are built;
3. each final image contains the expected LoRA path and exact SHA;
4. selected baked H3 LoRA produces a validated loader chain in each supported
   workflow family;
5. selected baked Krea LoRA resolves without an R2 download;
6. R2-backed LoRAs continue to work unchanged;
7. non-compatible LoRAs cannot be sent to a runtime;
8. a partial target failure never changes D1 to `baked`;
9. the runtime reports LoRA UUID, source, SHA, file path, and image digest in
   safe job diagnostics;
10. a provider canary succeeds on every rebuilt final image before `latest` is
    advanced.

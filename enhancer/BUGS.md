# Enhancer production checklist

This is the handoff/checklist for SceneBuilder2 + GPU-runtime Enhancer work. H3 generation logic is working and must not be modified to solve Enhancer issues.

## Current state

- SceneBuilder2 PR #99 is merged.
- GPU-runtime PR #12 is merged.
- No H3 implementation files were intentionally changed by those Enhancer fixes.

## Admin / user behavior

- Enhancer Admin and H3 Admin controls must only be visible to the existing allowed admin emails, same policy as H3.
- Normal users do not see admin controls and use saved/default settings.
- Enhancer video encoder default for normal users: `NVENC`, `CQ 17`.
- Admin may override encoder to exactly one of:
  - `NVENC` -> `CQ` only, range `12-25`.
  - `x265/libx265` -> `CRF` only, range `12-25`.
- Keep both saved values when switching encoder; only the selected encoder is active for a job.
- Every video job must carry `videoEncoder`, `nvencCq`, and `x265Crf`; pod runtime must use job settings, not hidden Docker/env overrides.
- When Director Controls tab is `Upscale`, per-clip/timeline primary actions must say `Upscale/Re-upscale`, not `Generate/Regenerate`.
- Bulk Upscale Selected/All must use a proper modal like Generate Selected/All with explicit `remaining` vs `re-upscale all` choice.

## Encoder/runtime

- FAST and QUALITY images must contain FFmpeg with both `hevc_nvenc` and `libx265`.
- Pod boot performs base GPU/CUDA qualification only.
- NVENC hardware smoke runs only for a job that selected NVENC; x265 jobs must not fail because NVENC is unavailable.
- NVENC uses HEVC Main10/p010 + CQ; x265 uses HEVC 10-bit + CRF.
- Keep GPU enhancement (ESRGAN/RIFE/FlashVSR/TensorRT) on GPU regardless of final encoder choice.
- Previous failure fixed: invalid 128x128 NVENC HEVC smoke caused FFmpeg exit status 234. Use a valid 256x256 probe and include FFmpeg stderr/stdout in errors.

## Idle / warm-pod behavior — NEXT REQUIRED PARITY ITEM

- A worker must not start its 60s termination timer merely because one job finished if compatible queued work still exists.
- Before entering idle countdown, SceneBuilder should try to hand the worker another compatible queued job.
- This should include compatible FAST/QUALITY work from another user/project when safe, matching H3 warm-pool/lending behavior rather than permanently project-scoping an otherwise usable GPU.
- Engine-builder workers should likewise take the next compatible queued engine build before idle countdown.
- The 60s termination timer starts only after the worker is actually idle: no compatible queued job remains for that image/service/GPU capability.
- Do not delete a busy worker. Provider deletion retry is separate from GPU-job retry.
- Re-check current SceneBuilder warm-worker ownership/accounting before implementing cross-project lending so per-project max-8 accounting is not corrupted.

## Engine builder

- `.engine` builder uses FAST image.
- One healthy compatible builder should process the engine queue sequentially.
- Without Force rebuild: completed/active matching profiles are reused; failed/cancelled/missing profiles may be created again.
- With Force rebuild: intentionally rebuild requested profiles.
- Keep one resolved Docker image digest across a running engine batch so `latest` changing mid-batch does not switch builders.
- D1 idempotency unique index must not reserve failed/cancelled builds.

## Fresh image behavior

- Keep provider image reference as normal Docker tag (for example `:latest`) because RunPod/Novita expect tag-style image fields.
- Resolve Docker Hub `latest` manifest digest separately and store/compare it for warm-worker freshness.
- Force Fresh ON: reuse only a warm worker whose recorded runtime digest equals current Docker Hub digest; otherwise retire stale idle worker and start a new tagged pod.
- Public Docker Hub repos use anonymous short-lived registry pull token; no long-lived Cloudflare DockerHub secret required for digest lookup.
- Previous symptom: Fresh ON could create no pod when provider was given `repo@sha256:...`; Fresh OFF with `:latest` started correctly.

## Provider parity

- RunPod: follow H3 availability fallback `SECURE -> COMMUNITY`.
- Novita create payload should use H3-compatible fields (`kind`, `billingMode`, `autoRenew`, etc.).
- Novita instance status/delete verification must use `GET /gpu/instance?instanceId=...`; never the broken `/gpu/instance/{id}` path.
- Do not treat a transient Novita read failure/404 as proof the instance is gone.
- Previous errors: `PROVIDER_HTTP_404: 404 page not found`, `startup_queue_aborted`, `provider_not_found`, followed by immediate deletion.

## Callback / recovery behavior

- Pod callback auth shape matches H3: JSON + `Authorization: Bearer <pod token>`.
- Cloudflare Access has returned HTTP 200 sign-in HTML on `/api/projects/v2/h3/pod/events`; Enhancer exposed this as `HTTP 200 NON_JSON` while H3 treats any 2xx as callback success.
- Do not depend on callbacks as the sole source of truth. Keep H3-style direct pod polling/reconciliation (`/jobs/{id}` for Enhancer) to recover progress/completion.
- Callback/recovery events must be order-safe: duplicate `worker_ready`, stale `worker_idle`, late progress, duplicate terminal callbacks must not move state backward or clear an active assignment.
- GPU/job failures are terminal; do not rerun the same failed GPU job automatically. Provider deletion retry is allowed separately.

## D1 / control-plane invariants

- Enhancer schema bootstrap/compatibility must create/repair required tables, columns, indexes, dispatch locks, and config rows on Worker request/maintenance path even if a migration was missed.
- Known historical error fixed: `UNIQUE constraint failed: enhancer_engine_builds.idempotency_key` on Force rebuild.
- Normal job dispatch and engine dispatch must be serialized/leased to avoid duplicate pod provisioning.
- GPU-success + SceneBuilder writeback failure must leave job as `writeback_failed` but release the worker for more work; do not rerun GPU work.
- Original-video dedupe must remain based on full source identity + derivative settings, never clip trim/link state. GPU pipelines process the full original source file.
- Encoder choice and active CQ/CRF should be part of video derivative/dedupe identity so switching NVENC/x265 or quality cannot incorrectly reuse an old encode.

## Scaling

- Target roughly 30 seconds of unique original-video work per active pod, max 8 live pods per project, matching H3 intent.
- Cross-project warm-pod lending must not break project accounting or the per-project cap.
- Do not reintroduce MIG/partition rejection.

## Errors encountered

- D1 `SQLITE_CONSTRAINT_UNIQUE` on engine Force rebuild.
- RunPod Fresh ON: no pod when digest-form provider image was used.
- Novita `PROVIDER_HTTP_404: 404 page not found` -> `startup_queue_aborted` -> `provider_not_found` -> immediate delete.
- Cloudflare Access callback response: HTTP 200 sign-in HTML / `NON_JSON`.
- NVENC qualification failure: FFmpeg exit status 234 from 128x128 HEVC probe.
- Enhancer callbacks could be lost/out-of-order; recovery polling must remain authoritative backup.

## Deploy / build / smoke checklist

1. Confirm SceneBuilder main containing PR #99 is deployed by Cloudflare.
2. Rebuild and push Enhancer FAST + QUALITY `latest` from GPU-runtime main. Engine builder uses FAST.
3. Confirm Docker Hub digest changed and new pods report the new digest.
4. Smoke test RunPod and Novita separately.
5. Smoke test FAST ESRGAN, RIFE interpolation, QUALITY/FlashVSR, NVENC CQ, x265 CRF, engine resume without Force, and Force rebuild.
6. Next work pass: implement/verify admin-email visibility and H3-style cross-project warm-pod lending/idle timing above before calling Enhancer fully complete.

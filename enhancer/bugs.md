# Enhancer production handoff

## Goal
Make Enhancer behave like the working H3 pod system without changing H3 generation logic: durable D1 state, immediate dispatch, polling/reconciliation as recovery, safe provider lifecycle, reusable warm pods, and terminal GPU job failures.

## Included in current PRs
- Upscale Selected/All uses a real Remaining vs Re-upscale All modal; Upscale controls make per-clip Generate/Regenerate become Upscale/Re-upscale.
- FAST/QUALITY support one explicit video encoder per job: `nvenc` (HEVC NVENC + CQ) or `x265` (`libx265` + CRF). Defaults: NVENC CQ 17, x265 CRF 15. Runtime clamps 12-25.
- NVENC qualification is job-specific and uses a valid 256x256 HEVC Main10 probe; x265 jobs do not require NVENC.
- Fresh Image keeps provider image refs as normal tags, resolves Docker Hub `latest` digest for identity, and retires stale warm pods.
- RunPod provider fallback matches H3 SECURE -> COMMUNITY.
- Novita create/status/delete API shapes are aligned to H3 and transient status reads must not delete a healthy instance.
- New pods do not start the idle-expiry clock at `worker_ready`.
- Callback HTTP 2xx/non-JSON handling matches H3; direct pod polling remains recovery.

## Follow-up bugs / requirements after these PRs merge
1. **Admin visibility:** Enhancer Admin and H3 video-generation Admin UI must only render for allowed admin emails. Normal users use defaults only. Do not expose provider/GPU/encoder/engine controls to normal users.
2. **Encoder config contract:** SceneBuilder backend must clamp Admin NVENC CQ and x265 CRF to 12-25 too (not only UI/runtime). Keep defaults NVENC + CQ 17 and x265 CRF 15. Admin can override. Persist selected encoder in `enhancer_config.runtime_policy_json`.
3. **Encoder-aware dedupe:** video source dedupe/idempotency must include encoder and active quality value (NVENC CQ or x265 CRF), so switching encoder/quality cannot incorrectly reuse an older upscale. Preserve original-source dedupe: trims/link state must still not create duplicate GPU enhancement jobs.
4. **True H3-style idle retention:** 60-second termination countdown starts only after the GPU is truly idle. Before arming/deleting an idle FAST/QUALITY/builder pod, check D1 for compatible queued work across projects/users and lend/reuse the pod like H3. Consider both `pending_upscales` and `enhancer_engine_builds`. Do not kill a compatible pod while queued work exists.
5. **Build validation:** FAST and QUALITY Docker builds should fail if FFmpeg lacks `libx265`; NVENC capability is runtime/GPU-specific, so verify `hevc_nvenc` is compiled into FFmpeg but keep the real NVENC hardware smoke job-specific.
6. **Callback/Access:** Cloudflare Access currently returns HTTP 200 sign-in HTML to pod callbacks. H3 hides this because it accepts any 2xx; Enhancer polling recovers. Keep polling authoritative/reliable and, if possible, add a narrow machine-to-machine Access path for `/api/projects/v2/h3/pod/events` rather than making the Worker public.
7. **D1 schema:** keep runtime compatibility/bootstrap plus migrations for all Enhancer tables/columns/indexes. Verify production schema after deploy, especially engine idempotency partial index, dispatch locks, worker digest fields, config JSON, and pending-upscale columns.

## Production errors encountered
- `D1_ERROR: UNIQUE constraint failed: enhancer_engine_builds.idempotency_key` on repeated/forced engine builds.
- Novita `PROVIDER_HTTP_404: 404 page not found` -> false `provider_not_found` -> pod deleted immediately.
- Pod callbacks returned `HTTP 200 NON_JSON` with Cloudflare Access sign-in HTML instead of Worker JSON.
- Fresh Image ON could fail to provision when a digest-form image ref was sent to the provider; Fresh Image OFF with normal `:latest` started correctly.
- FFmpeg NVENC qualification failed with exit status 234 because the smoke test used 128x128 HEVC; TensorRT/ESRGAN/RIFE work itself was not the cause.
- Engine builder could continue only through recovery polling when callbacks were intercepted.
- Upscale Selected/All initially showed a tiny menu instead of the expected Generate-style modal; clip actions still showed Generate/Regenerate while Upscale controls were selected.

## Release order
1. Merge SceneBuilder PR #99 and GPU-runtime PR #12 only with green CI and no H3 implementation-file changes.
2. Deploy SceneBuilder Worker/main and verify D1 compatibility/bootstrap runs successfully.
3. Rebuild/push **FAST** and **QUALITY** Docker `latest` images from merged GPU-runtime main (engine builder uses FAST).
4. Smoke test: engine builder resume without Force Rebuild; RunPod + Novita; Fresh Image ON/OFF; NVENC CQ and x265 CRF; original-video dedupe; idle reuse across queued work; writeback and deletion.
5. Implement the follow-up items above in one controlled parity pass, not isolated production hotfixes.

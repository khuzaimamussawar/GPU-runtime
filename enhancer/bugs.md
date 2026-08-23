# Enhancer production handoff

H3 video generation is the parity reference. Do not change H3 implementation logic to solve Enhancer issues.

## Baseline
- SceneBuilder2 PR #99 merged.
- GPU-runtime PR #12 merged.
- SceneBuilder2 PR #100 is the coordinated final repo-level parity pass.
- GPU-runtime PR #13 pins the same runtime boundary and CI contract.
- SceneBuilder PR #97 was not merged with stale ancestry; its two-file `AMPERE_PLUS` routing fix was cleanly carried onto current SceneBuilder `main` in PR #100.

## Repo-level items closed in PR #100 / #13
- Admin visibility/defaults: Enhancer Admin and H3 advanced text-encoder/Sage/Spectrum controls are allowlisted-admin only. Normal users use safe defaults. Enhancer normal-user output encoder remains NVENC CQ 17.
- Encoder contract: NVENC CQ and x265 CRF are clamped to 12-25 through SceneBuilder and GPU runtime. Defaults are NVENC CQ 17 and x265 CRF 15. One encoder is active per job.
- Encoder-aware derivative identity: SceneBuilder persists the active encoder + quality signature (`nvenc:cq=N` or `x265:crf=N`) and compares it for `Upscale remaining`, while original-source grouping continues to ignore trim/linkage.
- Warm-pod parity: SceneBuilder can lend compatible FAST/QUALITY idle workers across Director projects/users with target-project duration-derived cap accounting and hard max 8; engine builders remain globally reusable and prioritized.
- Idle retention/reaping considers compatible queued Director work globally; busy workers are not idle-reaper candidates.
- Provider parity remains: RunPod normal tag refs + digest identity, SECURE -> COMMUNITY fallback; Novita H3-compatible create/status/delete paths and transient-read tolerance.
- FAST and QUALITY Dockerfiles already verify FFmpeg exposes both `hevc_nvenc` and `libx265`; real NVENC hardware qualification is job-specific.
- `enhancer/runtime-contract.json` is mirrored in SceneBuilder and GPU-runtime for port 8000, H3 callback path, service kinds, 60s idle default, max-8 video pool, and encoder names/defaults/ranges. GPU parity CI now watches and executes the contract test.

## Intentionally unchanged in this pass
- Do not change `enhancer/src/callbacks.py` callback transport/parsing yet.
- Do not change H3 runtime implementation files.
- Keep pod API port **8000**. It is the inbound HTTP service used for `/health`, `/ready`, `/jobs`, and provider proxy mapping; it is unrelated to Cloudflare Access intercepting the pod's outbound HTTPS callback.

## Callback / Access state
- The observed production callback response is Cloudflare Access HTTP 200 sign-in HTML on pod -> SceneBuilder callbacks.
- The FAST image that logged `HTTP 200 NON_JSON` was not rebuilt after the last GPU-runtime callback parity changes, so it is running older callback code. Rebuild the final Docker layers before diagnosing callback-client behavior again.
- Direct pod polling/reconciliation remains authoritative recovery.
- Production still needs a live Access check and, if desired, a narrow machine-to-machine policy for `/api/projects/v2/h3/pod/events` rather than making the Worker public.

## Deployment-only work remaining
1. Merge SceneBuilder2 PR #100 and GPU-runtime PR #13 after green parity checks.
2. Deploy SceneBuilder `main` and verify production D1 compatibility/bootstrap, especially engine idempotency partial index, dispatch locks, worker digest fields, config JSON, and pending-upscale columns.
3. Rebuild/push **FAST** and **QUALITY** Docker `latest` from merged GPU-runtime `main` (engine builder uses FAST).
4. Verify Docker Hub digest changed and new workers report the expected digest.
5. Smoke RunPod + Novita, Fresh Image ON/OFF, FAST ESRGAN, RIFE, QUALITY/FlashVSR, NVENC CQ, x265 CRF, engine resume, Force Rebuild, encoder-aware `Upscale remaining`, and cross-project warm-worker reuse.
6. Re-check callback logs only after rebuilt images are running; separately verify whether Cloudflare Access is allowing the machine-to-machine callback to reach the Worker.

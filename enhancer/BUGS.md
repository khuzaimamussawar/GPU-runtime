# Enhancer production handoff

H3 video generation remains the lifecycle/parity reference. Do not change H3 implementation logic to solve Enhancer-only issues.

## Repo-level items closed
- Admin visibility/defaults: H3 advanced text-encoder/Sage/Spectrum controls are allowlisted-admin only. Normal users use the safe defaults. Enhancer video output defaults to NVENC CQ 17.
- Video encoder contract: exactly one of NVENC/x265 is active per job; quality is clamped to 12-25; defaults are NVENC CQ 17 and x265 CRF 15.
- Derivative encoder identity: SceneBuilder persists `nvenc:cq=N` / `x265:crf=N` for diagnostics, provenance and encoder-aware derivative identity. **It does not decide Director `Upscale remaining`.**
- Director `Upscale remaining`: a clip is remaining only when it has no completed upscale. Any completed upscale is skipped regardless of encoder/CQ/CRF/backend. Active upscale jobs are skipped. `Re-upscale all` and explicit per-clip re-upscale may rerun completed clips.
- Director FAST/QUALITY warm-pod parity: compatible idle workers can lend across Director projects/users with target-project duration-derived accounting and hard max 8. Engine builders remain separate/globally reusable.
- Idle retention/reaping considers compatible queued Director work globally; busy workers are never idle-reaper candidates.
- Provider parity remains: RunPod normal tag refs + digest identity and SECURE -> COMMUNITY fallback; Novita H3-compatible create/status/delete behavior and transient-read tolerance.
- FAST and QUALITY Dockerfiles verify FFmpeg exposes both `hevc_nvenc` and `libx265`; real NVENC hardware qualification remains job-specific.
- Shared runtime contract remains port 8000, H3 callback ingress `/api/projects/v2/h3/pod/events`, 60s idle default, shared service-kind names, max-8 Director video pool, and NVENC/x265 defaults/ranges.

## Storyboard image upscale contract
- SceneBuilder persists one durable D1 `image_upscale` row per scene/image. A temporary `image_upscale_batch` is only a pod execution envelope; member rows remain individually durable.
- Frontend/control plane sends the original active generated/uploaded scene image URL plus intrinsic source `width`/`height`; thumbnail sources/dimensions are invalid.
- Only images with the exact same source W×H and the same processing/routing contract may share one pod batch. Different W×H must be separate jobs/batches.
- Absolute runtime batch maximum is 20 images. Runtime validates declared W×H and verifies the downloaded source dimensions before processing.
- Runtime VRAM safety tiers are conservative: unknown/<24 GB => max 5; 24-32 GB => max 10; >32 GB => max 20.
- SceneBuilder capacity is one Storyboard image pod per 20 logical active image jobs, hard max 5 pods per project. This is independent of the Director video max-8 duration controller.
- Storyboard workers remain project-scoped; Director video warm workers retain their cross-project lending behavior.
- The same existing Enhancer/H3-style callback, direct-poll recovery, minute recovery, idle timeout and worker reaping lifecycle applies to Storyboard image jobs.

## Callback / Access state
- Pod API port **8000** is still the inbound HTTP service for `/health`, `/ready`, `/jobs` and provider proxy mapping. It is unrelated to the pod's outbound HTTPS callback.
- The observed production pod -> SceneBuilder callback response was Cloudflare Access HTTP 200 sign-in HTML.
- The image that logged `HTTP 200 NON_JSON` was stale because the final Docker layers had not been rebuilt. Current callback behavior should only be judged after rebuilding FAST/QUALITY from current `main`.
- Direct pod polling/reconciliation remains authoritative recovery when callbacks are missed.
- Cloudflare Access policy is a live deployment concern; repository code does not prove that the callback path is allowed machine-to-machine.

## Deployment-only work remaining
1. Deploy merged SceneBuilder `main` and verify production D1 bootstrap/schema compatibility and lifecycle rows.
2. Rebuild/push **FAST** and **QUALITY** Docker `latest` from merged GPU-runtime `main` (engine builder uses FAST).
3. Verify Docker Hub digests/runtime revisions changed and new pods report the expected revision/digest.
4. Smoke RunPod + Novita, Fresh Image ON/OFF, FAST ESRGAN, Storyboard same-W×H batching and mixed-W×H splitting, 5/10/20 VRAM batch guards, RIFE, QUALITY/FlashVSR, NVENC CQ, x265 CRF, engine resume/Force Rebuild, Director `Upscale remaining`, and cross-project Director warm-worker reuse.
5. Re-check callback logs only after rebuilt images are running; separately verify whether Cloudflare Access allows the machine-to-machine callback to reach the Worker.

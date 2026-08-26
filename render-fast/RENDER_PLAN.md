# GPU Fast Render Plan

## Scope

Fast is a GPU-only FFmpeg renderer for timelines that do not require Remotion.
It does not use RIFE, ESRGAN, PyTorch, Chromium, or the enhancer images.
Quality remains the existing CPU/Remotion path.

## Fast Eligibility

SceneBuilder computes the render plan before it creates a render row or asks a
provider for capacity.

Fast is disabled in the render modal and rejected by the Worker when any clip
requires a Remotion route, including:

- motion or animated transforms
- effects or animated overlays
- transitions
- any future planner reason marked `remotion`

The response lists the blocking clip IDs and reasons. It must not create a GPU
worker, a provider pod, or a billable render attempt. The Worker repeats this
check so a stale browser cannot bypass it.

## Ownership

`SceneBuilder2` owns the render modal, eligibility planner, D1 records,
provider scheduling, signed media URLs, and provider API credentials.

`GPU-runtime/render-fast/` owns one lightweight GPU pod image and its Fast
FFmpeg service. It is a manual build target in the existing GPU-runtime
workflow, not a new repository and not an enhancer build.

## Pod Contract

Use the existing H3 authentication master secret and callback shape, but derive
a render-scoped token from the worker ID. The pod therefore never receives an
H3 workload token value.

```text
SCENEBUILDER_POD_TOKEN=<per-render-worker secret>
SCENEBUILDER_WORKER_ID=<render_pod_workers.id>
SCENEBUILDER_CONTROL_URL=https://scene-builder.snorvians.workers.dev/api/projects/v2/render-gpu/pod/events
SCENEBUILDER_IDLE_TIMEOUT_SECONDS=<scheduler-selected timeout>
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
```

SceneBuilder resolves the full callback URL from `RENDER_GPU_POD_CONTROL_URL`
when configured; otherwise it uses the render callback URL above. It is an
address, not a credential: the per-worker bearer token remains mandatory.

The Worker stores only a hash of `SCENEBUILDER_POD_TOKEN`. The pod authenticates
every event with `Authorization: Bearer <token>`. Provider API keys, permanent
R2 credentials, Hetzner credentials, and user credentials never enter the
image or pod environment.

Each dispatched render contains only job-scoped data:

- signed source GET URLs
- signed output PUT URL
- render settings and resolved timeline plan
- output dimensions, FPS, codec, and quality mapping

Pod events follow H3 lifecycle semantics:

```text
worker_ready -> probe_complete -> worker_busy -> job_progress
-> job_done or job_failed -> worker_idle -> idle_expired -> deleted
```

## Provider Control

The admin Fast settings provide a segmented provider choice:

```text
Auto | RunPod | Novita
```

- Auto: try a verified RunPod placement first. Only when no viable RunPod
  candidate can be provisioned does it try Novita.
- RunPod: never switches to Novita; the job waits for RunPod capacity.
- Novita: never switches to RunPod.

Novita candidates are RTX 4090 (24 GB), RTX 5090 (32 GB), and L40S (48 GB).
RunPod candidates are discovered from current provider availability and limited
to approved NVIDIA models from 16 GB through 48 GB.

## VRAM Eligibility

This is streaming FFmpeg work. Video duration does not accumulate VRAM. The
important constraints are decoded surfaces, CUDA filters, NVENC/NVDEC engine
throughput, and the measured realtime FPS for the requested output profile.

| Output profile | Minimum | Preferred | Notes |
| --- | --- | --- | --- |
| 720p/1080p Fast | 16 GB | 16-20 GB | RTX 2000 Ada and RTX A4000 remain eligible. |
| 2K/48 Fast | 16 GB | 20 GB | Requires its model/driver probe profile to pass. |
| 4K/24-30 Fast | 16 GB | 20-24 GB | 16 GB is allowed only after a passing 4K profile. |
| 4K/48 Fast | 20 GB | 24-48 GB | 16 GB is not selected by default; an explicit green 4K/48 benchmark may opt it in later. |

Keep the RTX 2000 Ada as an economy candidate for 1080p and 2K. Do not spend a
4K/48 job on it by default. The screenshot's 16-20 GB GPUs are cheap when they
are available, so they should not be globally excluded.

RunPod ranking is based on currently available, probe-verified candidates and
measured FPS per dollar, not VRAM alone. Expected candidates include RTX 2000
Ada, RTX A4000, RTX A4500, RTX 4000 Ada, RTX A5000, L4, RTX 4090, RTX 5090,
A40, L40/L40S, RTX A6000, and RTX 6000 Ada. AMD and unverified compute-only
cards are excluded.

## NVENC Probe And Retry

Every new pod must prove the selected render path before it receives user work:

1. `nvidia-smi` detects the assigned GPU and driver.
2. FFmpeg lists the required CUDA/NVENC/NVDEC components.
3. For an H.264 job, H.264 NVENC encode succeeds; for an H.265 job, HEVC
   Main10/P010 NVENC encode succeeds. It does not reject an H.265 render
   merely because H.264 is unavailable, or vice versa.
4. Hardware decode plus the selected-codec CUDA scale/FPS path succeeds.
5. The image's CUDA 13 userspace supports the pod architecture, including
   native Blackwell support. The provider host driver remains authoritative.
6. A short 1080p, 2K, and where eligible 4K/48 benchmark records realtime FPS
   and peak VRAM, GPU, NVENC, NVDEC, CPU, and disk usage.

If a probe fails, the worker becomes `nvenc_unavailable`, records the concrete
driver or FFmpeg error, is retired, and the job is requeued to another GPU.
Fast never falls back to x264/x265. The render fails only after the selected
provider policy has exhausted its viable candidates.

### Implemented CUDA Route

The Fast image bakes CUDA 13 and builds pinned FFmpeg with NVENC, NVDEC,
`scale_cuda`, and NPP support rather than relying on Ubuntu's packaged FFmpeg.
At pod boot it creates a tiny source using the selected output codec and proves
the real chain used by Fast:

```text
NVDEC -> scale_cuda (Lanczos cover) -> hwdownload ->
exact project-frame crop / speed / fps / trim -> final NVENC
```

The logs and `render_job_metrics.route` state whether the result is
`gpu_decode_cuda_scale_cpu_timing_nvenc` or `cpu_filter_nvenc_fallback`.
Crop positioning, speed, FPS conversion, trim, and final timeline ordering
deliberately remain in the established CPU filter section because the pinned FFmpeg route
does not have a verified CUDA equivalent for every one of those operations.
If a real source rejects hardware decoding or the CUDA graph, Fast restarts the
whole visual stream through the CPU-filter/NVENC path; it never mixes partial
GPU and CPU output in one file.

`RENDER_GPU_FILTER_MODE=cpu` is an emergency Worker-side switch that preserves
the CPU-filter/NVENC route without rebuilding the image. Its default is `auto`.

## Streaming Execution

Fast uses the existing `renders` table for the render job, settings, status,
progress, output URL, provider, and assigned worker. It does not introduce a
second render-job table. `render_pod_workers` only records reusable GPU-worker
state and measured capabilities.

For one timeline export, Fast starts one final H.264/HEVC NVENC encoder and
keeps it alive for the entire output. Each visual timeline unit is decoded,
trimmed, and normalized in timeline order, then streamed directly into that
same final encoder. The current unit's decoder/filter process and the final
encoder run concurrently through a bounded pipe.

```text
unit 1 decode/filter -> final encoder stdin -> final.mp4
unit 2 decode/filter -> same encoder stdin -> final.mp4
unit N decode/filter -> same encoder stdin -> final.mp4
```

Fast never decodes the entire video before encoding it. A five-minute 4K/48
timeline would require hundreds of GB of raw frames and add unnecessary disk
I/O and latency. There are no encoded per-clip MP4 intermediates. Source
downloads may be prefetched with a bounded one-unit lookahead, but output
frames remain timeline-ordered and bounded in memory.

## D1 Data Model

Fast extends the existing render records. It does not create a second
`render_jobs` or render-history table.

### `renders` (one row per render job)

`renders` remains the canonical user-visible history row for both Quality and
Fast. Existing fields continue to own the title, project, status, progress,
output URL, error, timestamps, and billed hours. Fast adds only durable job
provenance:

```text
execution_mode          quality | fast
provider_preference     auto | runpod | novita
gpu_worker_id           render_pod_workers.id, nullable until assigned
minimum_vram_mb         scheduler request floor for this output profile
actual_gpu_name         probe result for the worker that ran the job
actual_gpu_vram_mb      probe result
actual_driver_version   probe result
actual_encoder          h264_nvenc | hevc_nvenc
encoder_settings_json   resolved codec, cq, preset, pixel format, FPS, size
render_plan_json        resolved visual/audio plan and Fast eligibility result
attempt_count           retry accounting
next_retry_at           nullable retry schedule
last_error_code         stable scheduling/probe/render error code
```

`provider` records the provider actually selected after assignment. Legacy
`server_id` remains the Hetzner Quality-server reference; Fast uses the new
`gpu_worker_id` rather than overloading it. `render_plan_json` is separate
from the temporary dispatch payload so the completed row retains enough
provenance to diagnose a result later without retaining signed URLs or tokens.

Fast job status lives on this same row:

```text
queued -> waiting_for_gpu -> provisioning -> probing -> rendering
-> completed | failed | cancelled
```

### `render_pod_workers` (live provider-pod state)

This is deliberately separate from the existing IP-oriented `render_servers`
table. It describes a reusable RunPod or Novita GPU pod, not a user export.

```text
id                         primary key
provider                   runpod | novita
provider_pod_id            unique provider pod identifier
image, region
gpu_name, gpu_vram_mb, compute_capability, driver_version, ffmpeg_version
status                     provisioning | probing | idle | busy | draining |
                           nvenc_unavailable | delete_failed | deleted
active_jobs_json           assigned render IDs; supports calibrated parallelism
active_slots, max_slots    scheduler state for the current profile capacity
capabilities_json          exact probe results and supported profiles
pod_token_hash             hash only; the raw per-worker token is never stored
created_at, last_activity_at, idle_since, terminate_after
last_error_code, last_error
```

`busy` is derived from `active_jobs_json`, not a stale single-job flag. A pod
receives a new job only when it has an available calibrated slot. `deleted` and
`delete_failed` never count as capacity. A `delete_failed` row is retained only
until the reaper has confirmed provider deletion; it cannot block a fresh pod
for a project or profile.

### H3-Style Dispatch And Recovery

Fast follows the H3 scheduler pattern rather than batch/cohort failure logic:

1. Claim one queued render atomically for one available worker slot.
2. The pod reports `worker_busy(jobId)` before work, and its callbacks are
   idempotent for that exact job ID.
3. On `job_done`, `job_failed`, or a confirmed cancellation, release only that
   slot. Immediately assign the next eligible queued render; do not wait for a
   cron tick or an idle timeout.
4. Prefer queued work from the same project while it exists, then lend the
   idle slot to another eligible project. Project caps count only live,
   non-draining pod-worker slots for the requested Fast profile.
5. A `409 worker busy`, stale callback, or callback for another job never fails
   a batch. Reconcile that worker's `active_jobs_json` and retry/requeue only
   the unstarted job.
6. A lost or draining pod requeues each of its still-active jobs immediately,
   marks the worker for retirement, and provisions/reuses another eligible
   worker. It never waits for an idle timeout when queued work exists.
7. Cancelling a render sends a job-specific cancel request to the pod, stops
   its FFmpeg process, clears the slot after acknowledgement (or a bounded
   recovery timeout), and returns the pod to `idle` or its normal delete path.

The capacity calculation ignores `deleted` and `delete_failed` rows exactly as
the enhancer/H3 worker logic does. It counts only available, probe-verified,
non-draining slots of the requested render profile.

### `render_gpu_capability_profiles` (reusable calibration)

This small table prevents a blind retry on every new pod while keeping the
actual pod probe authoritative. Its key is GPU model plus driver, FFmpeg build,
and output profile, for example `L4|driver-XXX|ffmpeg-XXX|2160p-48-hevc`.

```text
gpu_fingerprint, output_profile, codec
nvenc_h264_ok, nvenc_hevc_main10_ok, nvdec_ok, cuda_filter_ok
benchmark_fps, peak_vram_mb, peak_gpu_percent
peak_nvenc_percent, peak_nvdec_percent, recommended_slots
failure_code, failure_detail, updated_at
```

The scheduler may use a green profile to rank candidates, but a new pod still
runs its short smoke/probe before it receives a user export. A failed driver
or NVENC probe marks that fingerprint unhealthy and sends the queued render to
the next allowed candidate; it never falls back to CPU encoding.

### `render_job_metrics` (existing telemetry table)

The existing metrics table remains the per-clip and overall telemetry stream.
Fast adds queryable GPU fields rather than burying every value in JSON:

```text
gpu_vram_used_mb, gpu_vram_total_mb, gpu_util_percent
gpu_encoder_util_percent, gpu_decoder_util_percent
render_fps, output_frames, expected_frames
```

Its existing `phase`, `clip_index`, `route`, CPU, memory, disk, elapsed time,
and `details_json` fields continue to record decode/filter/final-encode and
overall measurements. The dashboard can therefore show current progress and
peak CPU, RAM, disk, GPU, VRAM, NVENC, and NVDEC for an individual render.

### Retention And Reaping

The existing 15-minute cron owns cleanup. At each run it deletes terminal
non-history D1 rows whose last update is older than one hour:

```text
render_pod_workers:  deleted | nvenc_unavailable, or delete_failed after the
                     reaper has confirmed provider deletion
render_job_metrics:  terminal-job telemetry older than one hour
```

`renders` is never deleted by this cron, regardless of whether it is completed,
failed, cancelled, or deleted. The final R2 output object also follows a
separate storage-retention policy. Active, idle, provisioning, probing, busy,
draining, and unresolved `delete_failed` workers are never silently removed.
The explicit reaper owns their provider deletion and retry path, so a stuck
provider row cannot hide a billable pod or block replacement capacity.

## Parallel Scheduling

One GPU pod is not permanently limited to one job. Parallel work is permitted
only after the GPU/model/driver/profile benchmark has calibrated it.

The scheduler uses profile slots, not a blind VRAM target. It considers:

- peak VRAM, capped at 90 percent to retain a 10 percent safety reserve
- GPU compute utilization, allowed to reach 100 percent
- NVENC and NVDEC engine utilization, with realtime FPS deciding whether they
  have enough headroom for another job
- per-job achieved FPS, which must remain above the output FPS with headroom
- host CPU, RAM, disk, and active encoder session limits

The initial concurrency matrix is conservative:

| GPU VRAM | 4K/48 | 2K/48 | 1080p/48 |
| --- | --- | --- | --- |
| 16-20 GB | 1 | 1 | 1, then benchmark 2 |
| 24 GB | 1 | 1-2 after benchmark | 1-2 after benchmark |
| 32 GB | 1-2 after benchmark | 2 after benchmark | 2-3 after benchmark |
| 48 GB | 1-2 after benchmark | 2-3 after benchmark | 3-4 after benchmark |

The benchmark is authoritative. If adding a second job drops either job below
realtime or exceeds the VRAM ceiling, the scheduler lowers that profile's slot
count. It does not launch parallel jobs merely to fill unused VRAM: NVENC can
be the limiting engine while VRAM remains mostly free.

## Quality Mapping

Fast preserves the chosen container, resolution, aspect ratio, FPS, and codec.
H.264 remains 8-bit. HEVC remains Main10/P010. NVENC uses `-cq` with VBR rather
than CPU `-crf`; the resolved NVENC quality is persisted with the render so the
dashboard and logs show the actual encoder settings.

## Acceptance Gate

Before enabling Fast for users, compare it with Quality across shared URLs,
partial trims, gaps, overlaps, speed, crop/scale, original versus upscale
selection, audio mute/gain/LUFS/keep ranges, 1080p/2K/4K, portrait/landscape,
H.264/H.265, and 24/30/48/60 FPS. Require exact duration within one frame,
monotonic timestamps, correct cut frames, audio sync, and successful playback
in VLC and browser playback.

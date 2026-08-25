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

Use the existing H3 environment and callback shape, but issue a render-scoped
token. Do not reuse an H3 token value.

```text
SCENEBUILDER_POD_TOKEN=<per-render-worker secret>
SCENEBUILDER_WORKER_ID=<render_gpu_workers.id>
SCENEBUILDER_CONTROL_URL=https://scene-builder.snorvians.workers.dev/api/projects/v2/render-gpu/pod/events
SCENEBUILDER_IDLE_TIMEOUT_SECONDS=<scheduler-selected timeout>
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
```

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

Every new pod must prove all of the following before it receives user work:

1. `nvidia-smi` detects the assigned GPU and driver.
2. FFmpeg lists the required CUDA/NVENC/NVDEC components.
3. H.264 NVENC encode succeeds.
4. HEVC Main10/P010 NVENC encode succeeds.
5. Hardware decode plus the requested scale/FPS path succeeds.
6. A short 1080p, 2K, and where eligible 4K/48 benchmark records realtime FPS
   and peak VRAM, GPU, NVENC, NVDEC, CPU, and disk usage.

If a probe fails, the worker becomes `nvenc_unavailable`, records the concrete
driver or FFmpeg error, is retired, and the job is requeued to another GPU.
Fast never falls back to x264/x265. The render fails only after the selected
provider policy has exhausted its viable candidates.

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

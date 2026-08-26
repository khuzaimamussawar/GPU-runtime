from __future__ import annotations

import json
import math
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable


HOST = os.environ.get("RENDER_GPU_HOST", "0.0.0.0")
PORT = int(os.environ.get("RENDER_GPU_PORT", "8000"))
POD_TOKEN = os.environ.get("SCENEBUILDER_POD_TOKEN", "").strip()
WORKER_ID = os.environ.get("SCENEBUILDER_WORKER_ID", "").strip()
CONTROL_URL = os.environ.get("SCENEBUILDER_CONTROL_URL", "").strip()
DEFAULT_IDLE_TIMEOUT = max(0, int(os.environ.get("SCENEBUILDER_IDLE_TIMEOUT_SECONDS", "60")))
REQUIRED_CODEC = os.environ.get("SCENEBUILDER_REQUIRED_CODEC", "").strip().lower()
FFMPEG_FLAVOR = os.environ.get("SCENEBUILDER_FFMPEG_FLAVOR", "system-ffmpeg").strip()
MAX_ERROR_DETAIL_CHARS = 16000
MAX_CONCURRENT_JOBS = max(1, min(4, int(os.environ.get("RENDER_GPU_MAX_CONCURRENT_JOBS", "4"))))
FRAME_BLOCK_FRAMES = max(1, min(16, int(os.environ.get("RENDER_GPU_FRAME_BLOCK_FRAMES", "8"))))
LOOKAHEAD_BUFFER_SECONDS = max(1, min(4, int(os.environ.get("RENDER_GPU_LOOKAHEAD_BUFFER_SECONDS", "2"))))
SYSTEM_MEMORY_CEILING_PERCENT = max(80.0, min(95.0, float(os.environ.get("RENDER_GPU_SYSTEM_MEMORY_CEILING_PERCENT", "90"))))


class RenderError(RuntimeError):
    pass


class GpuPipelineError(RenderError):
    pass


class OrderedFrameBuffer:
    """Bounded per-clip raw-frame queues consumed strictly in timeline order."""

    def __init__(self, clip_count: int, max_bytes_per_clip: int) -> None:
        self._condition = threading.Condition()
        self._queues: dict[int, deque[bytes]] = {index: deque() for index in range(clip_count)}
        self._queued_bytes: dict[int, int] = {index: 0 for index in range(clip_count)}
        self._closed: set[int] = set()
        self._errors: dict[int, str] = {}
        self._max_bytes_per_clip = max(1, max_bytes_per_clip)

    def put(self, index: int, block: bytes, should_stop: Callable[[], None]) -> None:
        while True:
            wait_for_system_memory_headroom(should_stop)
            with self._condition:
                if self._queued_bytes[index] + len(block) <= self._max_bytes_per_clip:
                    should_stop()
                    self._queues[index].append(block)
                    self._queued_bytes[index] += len(block)
                    self._condition.notify_all()
                    return
                should_stop()
                self._condition.wait(timeout=0.2)

    def close(self, index: int, error: Exception | str | None = None) -> None:
        with self._condition:
            self._closed.add(index)
            if error:
                self._errors[index] = str(error)
            self._condition.notify_all()

    def take(self, index: int, should_stop: Callable[[], None]) -> bytes | None:
        with self._condition:
            while not self._queues[index] and index not in self._closed:
                should_stop()
                self._condition.wait(timeout=0.2)
            should_stop()
            if not self._queues[index]:
                return None
            block = self._queues[index].popleft()
            self._queued_bytes[index] -= len(block)
            self._condition.notify_all()
            return block

    def error(self, index: int) -> str | None:
        with self._condition:
            return self._errors.get(index)


def parallel_clip_worker_count(settings: dict[str, Any]) -> int:
    """Scale the approved profile table to any provider vCPU allocation."""
    vcpus = max(1, int(settings.get("_physicalVcpus") or os.cpu_count() or 1))
    cpu_budget = max(1, int(settings.get("_cpuBudget") or vcpus))
    # Each 4K/48 preparation worker receives roughly three vCPUs. This makes
    # the 6/9/12/16 reference rows exact while accommodating 8, 10, 14, 20,
    # and other provider allocations without a separate hardcoded row.
    base_workers = max(1, math.ceil(cpu_budget / 3))

    width = max(2, int(settings.get("width") or 1920))
    height = max(2, int(settings.get("height") or 1080))
    fps = max(1, int(settings.get("fps") or 30))
    pixels = width * height
    if pixels >= 3840 * 2160:
        multiplier = 48 / fps
    elif pixels >= 2560 * 1440:
        multiplier = 2 if fps <= 48 else 1.8
    else:
        multiplier = 4 if fps <= 48 else 3.2

    return max(1, min(12, vcpus, int(math.floor(base_workers * multiplier))))


class State:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.work_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=MAX_CONCURRENT_JOBS)
        self.active_job_ids: set[str] = set()
        self.active_processes: dict[str, list[subprocess.Popen[bytes]]] = {}
        self.max_concurrent_jobs = 1
        self.capabilities: dict[str, Any] = {}
        self.status = "starting"
        self.idle_since: float | None = None
        self.terminate_after: float | None = None
        self.draining = False


STATE = State()


def _callback(event: str, *, attempts: int = 1, log_delivery: bool = False, **fields: Any) -> None:
    if not CONTROL_URL:
        return
    payload = {"event": event, "workerId": WORKER_ID, "timestamp": int(time.time() * 1000), **fields}
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": "SceneBuilder-GPU-Render/1.0"}
    if POD_TOKEN:
        headers["Authorization"] = f"Bearer {POD_TOKEN}"
    request = urllib.request.Request(CONTROL_URL, data=body, headers=headers, method="POST")
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            with urllib.request.urlopen(request, timeout=12) as response:
                response.read(1024)
            if log_delivery:
                print(f"[GPU Render] callback {event}: delivered on attempt {attempt + 1}", flush=True)
            return
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max(1, attempts):
                print(f"[GPU Render] callback {event}: retrying after {type(exc).__name__}: {exc}", flush=True)
                time.sleep(min(8, 2 ** attempt))
    print(f"[GPU Render] callback {event} failed: {type(last_error).__name__}: {last_error}", flush=True)


def emit(event: str, **fields: Any) -> None:
    threading.Thread(target=_callback, args=(event,), kwargs=fields, daemon=True).start()


def emit_startup_events(capabilities: dict[str, Any]) -> None:
    def announce() -> None:
        # These events are idempotent. Retry only during boot, when provider networking can lag the process.
        _callback("worker_ready", attempts=5, log_delivery=True)
        _callback("probe_complete", attempts=5, log_delivery=True, capabilities=capabilities)
    threading.Thread(target=announce, name="gpu-render-startup-callback", daemon=True).start()


def run(command: list[str], label: str, stdout: Any = None) -> subprocess.CompletedProcess[bytes]:
    print(f"[GPU Render] {label}: {' '.join(command[:12])}", flush=True)
    completed = subprocess.run(command, stdout=stdout, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace")[-MAX_ERROR_DETAIL_CHARS:]
        raise RenderError(f"{label} exited with {completed.returncode}: {error}")
    print(f"[GPU Render] {label}: passed", flush=True)
    return completed


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, timeout=15, text=True).strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def gpu_stats() -> dict[str, Any]:
    output = command_output([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,utilization.encoder,utilization.decoder",
        "--format=csv,noheader,nounits",
    ])
    first = output.splitlines()[0] if output else ""
    parts = [value.strip() for value in first.split(",")]
    if len(parts) < 7:
        return {"available": False, "error": output}
    def number(value: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0
    capability_lines = command_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"]).splitlines()
    capability = capability_lines[0] if capability_lines else ""
    return {
        "available": True,
        "name": parts[0],
        "driverVersion": parts[1],
        "vramMb": int(number(parts[2])),
        "vramUsedMb": int(number(parts[3])),
        "gpuUtilPercent": number(parts[4]),
        "encoderUtilPercent": number(parts[5]),
        "decoderUtilPercent": number(parts[6]),
        "computeCapability": capability,
    }


def merge_gpu_peak(peak: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    """Keep the per-render high-water marks used by the central scheduler."""
    if not sample.get("available"):
        return peak
    peak["vramMb"] = max(int(peak.get("vramMb") or 0), int(sample.get("vramMb") or 0))
    peak["vramUsedMb"] = max(int(peak.get("vramUsedMb") or 0), int(sample.get("vramUsedMb") or 0))
    peak["gpuUtilPercent"] = max(float(peak.get("gpuUtilPercent") or 0), float(sample.get("gpuUtilPercent") or 0))
    peak["encoderUtilPercent"] = max(float(peak.get("encoderUtilPercent") or 0), float(sample.get("encoderUtilPercent") or 0))
    peak["decoderUtilPercent"] = max(float(peak.get("decoderUtilPercent") or 0), float(sample.get("decoderUtilPercent") or 0))
    return peak


def disk_stats() -> dict[str, float]:
    usage = shutil.disk_usage("/")
    return {"diskUsedMb": round((usage.total - usage.free) / 1048576, 2), "diskFreeMb": round(usage.free / 1048576, 2)}


def host_cpu_percent() -> float:
    cores = max(1, os.cpu_count() or 1)
    try:
        return round(min(100.0, max(0.0, os.getloadavg()[0] * 100 / cores)), 2)
    except (AttributeError, OSError):
        return 0.0


def system_memory_stats() -> dict[str, float | bool]:
    """Read the pod's cgroup budget first, then use host memory as a fallback."""
    try:
        limit_text = Path("/sys/fs/cgroup/memory.max").read_text(encoding="utf-8").strip()
        current_text = Path("/sys/fs/cgroup/memory.current").read_text(encoding="utf-8").strip()
        if limit_text != "max":
            total = int(limit_text)
            used = int(current_text)
            if total > 0:
                return {
                    "systemMemoryAvailable": True,
                    "systemMemoryTotalMb": round(total / 1048576, 2),
                    "systemMemoryUsedMb": round(used / 1048576, 2),
                    "systemMemoryPercent": round(used * 100 / total, 2),
                }
    except (OSError, ValueError):
        pass
    try:
        total = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
        available = os.sysconf("SC_AVPHYS_PAGES")
        used = total - int(os.sysconf("SC_PAGE_SIZE")) * int(available)
        if total > 0:
            return {
                "systemMemoryAvailable": True,
                "systemMemoryTotalMb": round(total / 1048576, 2),
                "systemMemoryUsedMb": round(used / 1048576, 2),
                "systemMemoryPercent": round(used * 100 / total, 2),
            }
    except (AttributeError, OSError, ValueError):
        pass
    return {"systemMemoryAvailable": False}


def wait_for_system_memory_headroom(should_stop: Callable[[], None]) -> None:
    """Pause producers before buffered raw frames can exceed the pod RAM ceiling."""
    while True:
        memory = system_memory_stats()
        if not memory.get("systemMemoryAvailable") or float(memory.get("systemMemoryPercent") or 0) < SYSTEM_MEMORY_CEILING_PERCENT:
            return
        should_stop()
        time.sleep(0.2)


def ffmpeg_encoders() -> str:
    return command_output(["ffmpeg", "-hide_banner", "-encoders"])


def probe() -> dict[str, Any]:
    gpu = gpu_stats()
    encoders = ffmpeg_encoders()
    h264_listed = "h264_nvenc" in encoders
    hevc_listed = "hevc_nvenc" in encoders
    capabilities: dict[str, Any] = {
        "gpu": gpu,
        "ffmpegVersion": command_output(["ffmpeg", "-version"]).splitlines()[0],
        "encoders": {"h264Nvenc": False, "hevcNvencMain10": False},
        "profiles": [],
    }
    if not gpu.get("available"):
        capabilities["error"] = "nvidia-smi is unavailable"
        return capabilities
    with tempfile.TemporaryDirectory(prefix="sb-render-probe-") as work:
        h264_path = str(Path(work) / "h264.mp4")
        h265_path = str(Path(work) / "h265.mp4")
        if REQUIRED_CODEC in ("", "h264") and h264_listed:
            try:
                run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=256x256:r=1", "-frames:v", "1", "-c:v", "h264_nvenc", "-preset", "p4", h264_path], "H.264 NVENC smoke")
                capabilities["encoders"]["h264Nvenc"] = True
            except Exception as exc:
                print(f"[GPU Render] H.264 NVENC smoke: failed: {exc}", flush=True)
                capabilities["error"] = str(exc)
        if REQUIRED_CODEC in ("", "h265") and hevc_listed:
            try:
                run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=256x256:r=1", "-frames:v", "1", "-c:v", "hevc_nvenc", "-profile:v", "main10", "-pix_fmt", "p010le", "-preset", "p4", h265_path], "HEVC NVENC smoke")
                capabilities["encoders"]["hevcNvencMain10"] = True
            except Exception as exc:
                print(f"[GPU Render] HEVC NVENC smoke: failed: {exc}", flush=True)
                capabilities["hevcError"] = str(exc)
    for key, width, height, fps in (("1080p-48", 1920, 1080, 48), ("2k-48", 2560, 1440, 48), ("4k-48", 3840, 2160, 48)):
        if key == "4k-48" and int(gpu.get("vramMb") or 0) < 20000:
            continue
        capabilities["profiles"].append({"key": key, "codec": "h264", "fps": 0, "recommendedSlots": 1})
    return capabilities


def gpu_pipeline_for_clip(clip: dict[str, Any], use_gpu_pipeline: bool) -> bool:
    return use_gpu_pipeline and str(clip.get("type") or "video") == "video" and bool(str(clip.get("url") or "").strip())


def ffmpeg_video_decoder_args(clip: dict[str, Any], settings: dict[str, Any], use_gpu_pipeline: bool = False) -> list[str]:
    duration = max(0.001, float(clip.get("sceneDuration") or 0.001))
    media_type = str(clip.get("type") or "placeholder")
    source = str(clip.get("url") or "")
    threads = str(max(1, int(settings.get("_ffmpegThreads") or 1)))
    if media_type == "placeholder" or not source:
        return ["-threads", threads, "-f", "lavfi", "-i", f"color=c=black:s={settings['width']}x{settings['height']}:r={settings['fps']}", "-t", str(duration)]
    if media_type == "image":
        return ["-threads", threads, "-loop", "1", "-framerate", str(settings["fps"]), "-i", source, "-t", str(duration)]
    speed = max(0.1, float(clip.get("speed") or 1))
    source_duration = duration * speed
    start = max(0, float(clip.get("startTimeOffset") or 0))
    hardware_decode = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"] if gpu_pipeline_for_clip(clip, use_gpu_pipeline) else []
    return ["-threads", threads, "-accurate_seek", "-ss", str(start), "-t", str(source_duration), *hardware_decode, "-i", source]


def render_scale(value: Any) -> float:
    """Match TimelineRender's static zoom control while keeping an exact frame crop."""
    try:
        return max(1.0, min(3.0, float(value if value is not None else 1)))
    except (TypeError, ValueError):
        return 1.0


def even_dimension(value: float) -> int:
    return max(2, int(math.ceil(value / 2.0)) * 2)


def video_filter(clip: dict[str, Any], settings: dict[str, Any], use_gpu_pipeline: bool = False) -> str:
    duration = max(0.001, float(clip.get("sceneDuration") or 0.001))
    speed = max(0.1, float(clip.get("speed") or 1))
    width, height, fps = settings["width"], settings["height"], settings["fps"]
    scale = render_scale(clip.get("scale"))
    # Cover first, then crop to the exact project frame. Scaling the cover
    # target by the static zoom matches the Timeline preview's object-fit:cover
    # followed by its centered CSS scale transform.
    cover_width, cover_height = even_dimension(width * scale), even_dimension(height * scale)
    if gpu_pipeline_for_clip(clip, use_gpu_pipeline):
        filters = [
            f"scale_cuda={cover_width}:{cover_height}:force_original_aspect_ratio=increase:force_divisible_by=2:interp_algo=lanczos:format=nv12",
            "hwdownload",
            "format=nv12",
            f"crop={width}:{height}:(iw-ow)/2:(ih-oh)/2",
            "setsar=1",
        ]
    else:
        filters = [
            f"scale={cover_width}:{cover_height}:force_original_aspect_ratio=increase:flags=lanczos",
            f"crop={width}:{height}:(iw-ow)/2:(ih-oh)/2",
            "setsar=1",
        ]
    if str(clip.get("type") or "video") == "video" and speed != 1:
        filters.append(f"setpts=PTS/{speed}")
    filters.extend([f"fps={fps}", f"trim=duration={duration}", "setpts=PTS-STARTPTS", "format=yuv420p"])
    return ",".join(filters)


def audio_filter(clip: dict[str, Any]) -> str:
    filters: list[str] = []
    gain = clip.get("gainDb", clip.get("gain_db", 0))
    try:
        gain_number = float(gain)
        if gain_number:
            filters.append(f"volume={gain_number}dB")
    except (TypeError, ValueError):
        pass
    lufs = clip.get("lufs", clip.get("targetLufs"))
    try:
        if lufs is not None:
            filters.append(f"loudnorm=I={float(lufs)}:LRA=11:TP=-1.5")
    except (TypeError, ValueError):
        pass
    return ",".join(filters) if filters else "anull"


def render_audio_clip(clip: dict[str, Any], output_path: Path, work_dir: Path, settings: dict[str, Any]) -> None:
    duration = max(0.001, float(clip.get("sceneDuration") or 0.001))
    source = str(clip.get("audioSourceFile") or clip.get("url") or "")
    if bool(clip.get("isMuted")) or not source:
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=mono", "-t", str(duration), "-c:a", "pcm_s16le", str(output_path)], "audio silence")
        return
    base = float(clip.get("audioSourceStart") or 0) + float(clip.get("startTimeOffset") or 0)
    ranges = [item for item in (clip.get("keepRanges") or []) if float(item.get("end") or 0) > float(item.get("start") or 0)]
    filter_value = audio_filter(clip)
    if not ranges:
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(max(0, base)), "-i", source, "-t", str(duration), "-vn", "-af", f"{filter_value},apad", "-t", str(duration), "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(output_path)], "audio unit")
        return
    pieces: list[Path] = []
    for index, item in enumerate(ranges):
        start = float(item["start"])
        length = max(0.001, float(item["end"]) - start)
        piece = work_dir / f"audio-range-{output_path.stem}-{index}.wav"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(max(0, base + start)), "-i", source, "-t", str(length), "-vn", "-af", filter_value, "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(piece)], "retained audio range")
        pieces.append(piece)
    list_path = work_dir / f"{output_path.stem}.txt"
    list_path.write_text("\n".join(f"file '{path.as_posix()}'" for path in pieces), encoding="utf-8")
    joined = work_dir / f"{output_path.stem}-joined.wav"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c:a", "pcm_s16le", str(joined)], "join retained audio")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(joined), "-af", "apad", "-t", str(duration), "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(output_path)], "normalize retained audio")


def render_canonical_audio(audio_clips: list[dict[str, Any]], work_dir: Path, settings: dict[str, Any]) -> Path | None:
    if not audio_clips:
        return None
    units: list[Path] = []
    for index, clip in enumerate(audio_clips):
        output = work_dir / f"audio-{index:04d}.wav"
        render_audio_clip(clip, output, work_dir, settings)
        units.append(output)
    list_path = work_dir / "audio-units.txt"
    list_path.write_text("\n".join(f"file '{path.as_posix()}'" for path in units), encoding="utf-8")
    output = work_dir / "canonical-audio.wav"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(output)], "canonical audio mix")
    return output


def materialize_project_audio(job: dict[str, Any], work_dir: Path) -> Path | None:
    canonical = job.get("canonicalAudio") or {}
    source = str(canonical.get("url") or "").strip() if isinstance(canonical, dict) else ""
    if not source:
        return None
    downloaded = work_dir / "project-audio-source.wav"
    try:
        with urllib.request.urlopen(source, timeout=600) as response:
            downloaded.write_bytes(response.read())
    except Exception as exc:
        raise RenderError(f"project encoded audio download failed: {exc}") from exc
    output = work_dir / "canonical-audio.wav"
    total_seconds = max(0.001, float(job.get("durationInFrames") or 0) / max(1, int(job["settings"].get("fps") or 30)))
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(downloaded), "-af", "apad", "-t", str(total_seconds), "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(output)], "prepare project encoded audio")
    return output


def register_process(job_id: str, process: subprocess.Popen[bytes]) -> None:
    with STATE.lock:
        STATE.active_processes.setdefault(job_id, []).append(process)


def unregister_process(job_id: str, process: subprocess.Popen[bytes]) -> None:
    with STATE.lock:
        processes = STATE.active_processes.get(job_id, [])
        if process in processes:
            processes.remove(process)


def raise_if_cancelled(job_id: str) -> None:
    with STATE.lock:
        if STATE.jobs.get(job_id, {}).get("cancelRequested"):
            raise RenderError("RENDER_CANCELLED")


def terminate_registered_processes(job_id: str) -> None:
    with STATE.lock:
        processes = list(STATE.active_processes.get(job_id, []))
    for process in processes:
        if process.poll() is None:
            process.terminate()


def prepare_visual_unit(
    job: dict[str, Any],
    index: int,
    clip: dict[str, Any],
    settings: dict[str, Any],
    frame_bytes: int,
    frames_per_block: int,
    frame_buffer: OrderedFrameBuffer,
    abort_event: threading.Event,
    worker_number: int,
) -> None:
    """Run one CPU FFmpeg unit and publish bounded raw-frame blocks."""
    job_id = job["jobId"]
    expected_frames = max(1, round(max(0.001, float(clip.get("sceneDuration") or 0.001)) * settings["fps"]))
    decoder_threads = str(max(1, int(settings.get("_ffmpegThreads") or 1)))
    started = time.monotonic()
    output_frames = 0

    def should_stop() -> None:
        if abort_event.is_set():
            raise RenderError("VISUAL_PIPELINE_ABORTED")
        raise_if_cancelled(job_id)

    decoder: subprocess.Popen[bytes] | None = None
    try:
        # The rawvideo muxer otherwise preserves a source's native cadence on
        # some builds. Make the unit output contract explicit: one CFR frame
        # for every frame reserved by this timeline clip. The single-frame
        # clone pad covers clips whose requested end lands between source PTS
        # values; -frames:v remains the authoritative exact duration limit.
        unit_filter = f"{video_filter(clip, settings, False)},tpad=stop_mode=clone:stop_duration={1 / settings['fps']}"
        decoder = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-filter_threads", decoder_threads,
            *ffmpeg_video_decoder_args(clip, settings, False),
            "-map", "0:v:0", "-an", "-vf", unit_filter,
            "-fps_mode", "cfr", "-r", str(settings["fps"]), "-frames:v", str(expected_frames),
            "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1",
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        register_process(job_id, decoder)
        print(
            f"[GPU Render] {job_id} prepare clip={index} worker={worker_number} "
            f"expected_frames={expected_frames} decoder_threads={decoder_threads}",
            flush=True,
        )
        emit("job_metric", jobId=job_id, metric={
            "phase": "video_prepare_started", "clipIndex": index,
            "workerIndex": worker_number, "route": "cpu_filter_lanczos_nvenc",
            "details": {"decoderThreads": int(decoder_threads), "expectedFrames": expected_frames},
            "resource": {"cpuPercent": host_cpu_percent(), **disk_stats()},
        })
        assert decoder.stdout is not None
        while block := decoder.stdout.read(frame_bytes * frames_per_block):
            should_stop()
            if len(block) % frame_bytes:
                raise RenderError(f"visual unit {index} emitted an incomplete raw frame block")
            frame_buffer.put(index, block, should_stop)
            output_frames += len(block) // frame_bytes
        stderr = decoder.stderr.read().decode("utf-8", errors="replace") if decoder.stderr else ""
        if decoder.wait() != 0:
            raise RenderError(f"visual unit {index} failed: {stderr[-MAX_ERROR_DETAIL_CHARS:]}")
        if output_frames != expected_frames:
            raise RenderError(f"visual unit {index} produced {output_frames} frames; expected {expected_frames}")
        frame_buffer.close(index)
        elapsed = max(0.001, time.monotonic() - started)
        print(
            f"[GPU Render] {job_id} prepared clip={index} worker={worker_number} "
            f"frames={output_frames} fps={output_frames / elapsed:.2f}",
            flush=True,
        )
        emit("job_metric", jobId=job_id, metric={
            "phase": "video_prepare_complete", "clipIndex": index,
            "workerIndex": worker_number, "route": "cpu_filter_lanczos_nvenc",
            "renderFps": round(output_frames / elapsed, 2), "outputFrames": output_frames,
            "details": {"decoderThreads": int(decoder_threads), "expectedFrames": expected_frames},
            "resource": {"cpuPercent": host_cpu_percent(), **disk_stats()},
        })
    except Exception as exc:
        print(f"[GPU Render] {job_id} prepare failed clip={index} worker={worker_number}: {exc}", flush=True)
        frame_buffer.close(index, exc)
    finally:
        if decoder is not None and decoder.poll() is None:
            decoder.terminate()
            try:
                decoder.wait(timeout=10)
            except subprocess.TimeoutExpired:
                decoder.kill()
        if decoder is not None:
            unregister_process(job_id, decoder)


def render_video(job: dict[str, Any], audio: Path | None, work_dir: Path, use_gpu_pipeline: bool = False) -> tuple[Path, dict[str, Any]]:
    settings = job["settings"]
    clips = job["clips"]
    output = work_dir / "video.mp4"
    codec = str(settings.get("codec") or "h264")
    encoder_args = ["-c:v", "hevc_nvenc", "-profile:v", "main10", "-pix_fmt", "p010le", "-preset", str(settings.get("preset") or "p6"), "-rc", "vbr", "-cq", str(settings.get("cq") or 17), "-tag:v", "hvc1"] if codec == "h265" else ["-c:v", "h264_nvenc", "-preset", str(settings.get("preset") or "p6"), "-rc", "vbr", "-cq", str(settings.get("cq") or 17), "-pix_fmt", "yuv420p"]
    ffmpeg_threads = str(max(1, int(settings.get("_ffmpegThreads") or 1)))
    encoder = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", ffmpeg_threads, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-video_size", f"{settings['width']}x{settings['height']}", "-framerate", str(settings["fps"]), "-i", "pipe:0", "-map", "0:v:0", "-an", *encoder_args, "-r", str(settings["fps"]), str(output)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    register_process(job["jobId"], encoder)
    expected_frames = max(1, sum(max(1, round(max(0.001, float(clip.get("sceneDuration") or 0.001)) * settings["fps"])) for clip in clips))
    completed_frames = 0
    peak_gpu: dict[str, Any] = {}
    video_started = time.monotonic()
    render_route = "cpu_filter_lanczos_nvenc"
    frame_bytes = max(1, settings["width"] * settings["height"] * 3 // 2)
    parallel_workers = min(len(clips), max(1, int(settings.get("_parallelClipWorkers") or 1)))
    frames_per_block = min(max(1, int(settings.get("fps") or 1)), FRAME_BLOCK_FRAMES)
    buffered_bytes_per_clip = frame_bytes * max(1, int(settings.get("fps") or 1)) * LOOKAHEAD_BUFFER_SECONDS
    frame_buffer = OrderedFrameBuffer(len(clips), buffered_bytes_per_clip)
    abort_event = threading.Event()
    source_queue: queue.Queue[int] = queue.Queue()
    for index in range(len(clips)):
        source_queue.put(index)

    def worker_loop(worker_number: int) -> None:
        while not abort_event.is_set():
            try:
                index = source_queue.get_nowait()
            except queue.Empty:
                return
            try:
                prepare_visual_unit(job, index, clips[index], settings, frame_bytes, frames_per_block, frame_buffer, abort_event, worker_number)
            finally:
                source_queue.task_done()

    preparation_threads = [
        threading.Thread(target=worker_loop, args=(worker_number,), name=f"gpu-visual-prepare-{job['jobId'][:8]}-{worker_number}", daemon=True)
        for worker_number in range(parallel_workers)
    ]
    print(f"[GPU Render] {job['jobId']} visual route: {render_route}", flush=True)
    print(
        f"[GPU Render] {job['jobId']} visual scheduler: workers={parallel_workers} "
        f"decoder_threads={ffmpeg_threads} block_frames={frames_per_block} "
        f"lookahead_seconds={LOOKAHEAD_BUFFER_SECONDS}",
        flush=True,
    )
    emit("job_metric", jobId=job["jobId"], metric={
        "phase": "video_scheduler", "route": render_route,
        "details": {
            "parallelClipWorkers": parallel_workers,
            "decoderThreads": int(ffmpeg_threads),
            "frameBlockFrames": frames_per_block,
            "lookaheadBufferSeconds": LOOKAHEAD_BUFFER_SECONDS,
            "bufferedBytesPerClip": buffered_bytes_per_clip,
        },
        "resource": {"cpuPercent": host_cpu_percent(), **gpu_stats(), **disk_stats()},
    })
    try:
        merge_gpu_peak(peak_gpu, gpu_stats())
        for thread in preparation_threads:
            thread.start()
        for index, clip in enumerate(clips):
            raise_if_cancelled(job["jobId"])
            clip_route = "cpu_filter_lanczos_nvenc"
            unit_started = time.monotonic()
            unit_expected_frames = max(1, round(max(0.001, float(clip.get("sceneDuration") or 0.001)) * settings["fps"]))
            unit_frames = 0
            last_progress_emit = 0.0
            last_gpu_sample = time.monotonic()
            assert encoder.stdin is not None
            while block := frame_buffer.take(index, lambda: raise_if_cancelled(job["jobId"])):
                encoder.stdin.write(block)
                unit_frames += len(block) // frame_bytes
                elapsed = max(0.001, time.monotonic() - unit_started)
                if time.monotonic() - last_gpu_sample >= 1:
                    merge_gpu_peak(peak_gpu, gpu_stats())
                    last_gpu_sample = time.monotonic()
                if time.monotonic() - last_progress_emit >= 1:
                    completed = min(expected_frames, completed_frames + min(unit_expected_frames, unit_frames))
                    progress = 20 + int((completed / expected_frames) * 70)
                    emit("job_progress", jobId=job["jobId"], progress=progress, phase="video", clipIndex=index)
                    emit("job_metric", jobId=job["jobId"], metric={
                        "phase": "video", "clipIndex": index, "route": clip_route,
                        "renderFps": round(unit_frames / elapsed, 2), "outputFrames": completed, "expectedFrames": expected_frames,
                        "resource": {"cpuPercent": host_cpu_percent(), **gpu_stats(), **disk_stats()},
                    })
                    last_progress_emit = time.monotonic()
            if error := frame_buffer.error(index):
                raise RenderError(error)
            if unit_frames != unit_expected_frames:
                raise RenderError(f"visual unit {index} wrote {unit_frames} frames; expected {unit_expected_frames}")
            sample = gpu_stats()
            merge_gpu_peak(peak_gpu, sample)
            completed_frames += unit_expected_frames
            elapsed = max(0.001, time.monotonic() - unit_started)
            progress = 20 + int((min(expected_frames, completed_frames) / expected_frames) * 70)
            emit("job_progress", jobId=job["jobId"], progress=progress, phase="video", clipIndex=index)
            emit("job_metric", jobId=job["jobId"], metric={
                "phase": "video", "clipIndex": index, "route": clip_route,
                "renderFps": round(unit_expected_frames / elapsed, 2), "outputFrames": min(expected_frames, completed_frames), "expectedFrames": expected_frames,
                "resource": {"cpuPercent": host_cpu_percent(), **sample, **disk_stats()},
            })
        assert encoder.stdin is not None
        encoder.stdin.close()
        stderr = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
        if encoder.wait() != 0:
            message = f"final NVENC encode failed: {stderr[-MAX_ERROR_DETAIL_CHARS:]}"
            if use_gpu_pipeline:
                raise GpuPipelineError(message)
            raise RenderError(message)
        merge_gpu_peak(peak_gpu, gpu_stats())
    except BrokenPipeError as exc:
        raise RenderError(f"final encoder pipe failed: {exc}") from exc
    finally:
        abort_event.set()
        terminate_registered_processes(job["jobId"])
        for thread in preparation_threads:
            thread.join(timeout=10)
        if encoder.stdin is not None and not encoder.stdin.closed:
            try:
                encoder.stdin.close()
            except OSError:
                pass
        if encoder.poll() is None:
            encoder.terminate()
            try:
                encoder.wait(timeout=10)
            except subprocess.TimeoutExpired:
                encoder.kill()
        unregister_process(job["jobId"], encoder)
    peak_gpu["renderFps"] = round(expected_frames / max(0.001, time.monotonic() - video_started), 2)
    if audio is None:
        return output, peak_gpu
    muxed = work_dir / "output.mp4"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(output), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", str(settings.get("audioBitrate") or "192k"), "-ar", "48000", "-ac", "1", "-movflags", "+faststart", str(muxed)], "final audio mux")
    return muxed, peak_gpu


def upload(url: str, path: Path) -> None:
    request = urllib.request.Request(url, data=path.read_bytes(), headers={"Content-Type": "video/mp4"}, method="PUT")
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            response.read(1024)
    except urllib.error.HTTPError as exc:
        raise RenderError(f"output upload failed: HTTP {exc.code}") from exc


def process_job(job: dict[str, Any]) -> None:
    job_id = job["jobId"]
    started = time.time()
    emit("job_progress", jobId=job_id, progress=1, phase="starting")
    try:
        with tempfile.TemporaryDirectory(prefix=f"sb-render-{job_id[:8]}-") as temporary:
            work_dir = Path(temporary)
            audio = materialize_project_audio(job, work_dir) or render_canonical_audio(job["audioClips"], work_dir, job["settings"])
            emit("job_progress", jobId=job_id, progress=20, phase="audio_ready")
            # Keep every timing-sensitive visual filter on CPU, then send frames
            # once to the selected final NVENC encoder. Startup validates NVENC
            # only; CUDA filters are not a readiness or dispatch gate.
            use_gpu_pipeline = False
            selected_route = "cpu_filter_lanczos_nvenc"
            print(f"[GPU Render] {job_id} visual filters: CPU Lanczos; selected route: {selected_route}", flush=True)
            emit("job_metric", jobId=job_id, metric={
                "phase": "visual_route", "route": selected_route,
                "details": {"ffmpegFlavor": FFMPEG_FLAVOR},
                "resource": {"cpuPercent": host_cpu_percent(), **gpu_stats(), **disk_stats()},
            })
            try:
                output, peak_gpu = render_video(job, audio, work_dir, use_gpu_pipeline=use_gpu_pipeline)
            except GpuPipelineError as exc:
                # The CPU path is the established compatibility route. Restart the
                # entire visual stream, never splice CPU frames into a failed GPU run.
                print(f"[GPU Render] {job_id} CUDA pipeline rejected a source; restarting CPU fallback: {exc}", flush=True)
                emit("job_metric", jobId=job_id, metric={
                    "phase": "gpu_pipeline_fallback", "route": "cpu_filter_lanczos_nvenc",
                    "details": {"gpuPipelineError": str(exc)[-MAX_ERROR_DETAIL_CHARS:]},
                    "resource": {"cpuPercent": host_cpu_percent(), **gpu_stats(), **disk_stats()},
                })
                output, peak_gpu = render_video(job, audio, work_dir, use_gpu_pipeline=False)
            raise_if_cancelled(job_id)
            emit("job_progress", jobId=job_id, progress=95, phase="uploading")
            upload(str(job["uploadUrl"]), output)
            settings = job["settings"]
            emit(
                "job_done",
                jobId=job_id,
                outputUrl=job.get("outputUrl"),
                elapsedMs=int((time.time() - started) * 1000),
                calibration={
                    "outputProfile": f"{settings['width']}x{settings['height']}-{settings['fps']}",
                    "codec": str(settings.get("codec") or "h264"),
                    "peakVramMb": int(peak_gpu.get("vramUsedMb") or 0),
                    "totalVramMb": int(peak_gpu.get("vramMb") or 0),
                    "peakGpuPercent": float(peak_gpu.get("gpuUtilPercent") or 0),
                    "peakNvencPercent": float(peak_gpu.get("encoderUtilPercent") or 0),
                    "peakNvdecPercent": float(peak_gpu.get("decoderUtilPercent") or 0),
                    "renderFps": float(peak_gpu.get("renderFps") or 0),
                },
            )
    except Exception as exc:
        cancelled = str(exc) == "RENDER_CANCELLED"
        event = "job_cancelled" if cancelled else "job_failed"
        code = "RENDER_CANCELLED" if cancelled else ("GPU_OOM" if "out of memory" in str(exc).lower() else "GPU_RENDER_FAILED")
        emit(
            event,
            jobId=job_id,
            errorCode=code,
            error=str(exc),
            errorDetails={
                "exceptionType": type(exc).__name__,
                "gpu": gpu_stats(),
                "disk": disk_stats(),
            },
        )
    finally:
        idle_event: tuple[int, int] | None = None
        with STATE.lock:
            STATE.active_job_ids.discard(job_id)
            STATE.active_processes.pop(job_id, None)
            STATE.jobs.pop(job_id, None)
            if not STATE.active_job_ids:
                STATE.status = "idle"
                STATE.idle_since = time.time()
                timeout = int(job.get("idleTimeoutSeconds") or DEFAULT_IDLE_TIMEOUT)
                STATE.terminate_after = STATE.idle_since + timeout
                idle_event = (int(STATE.idle_since * 1000), int(STATE.terminate_after * 1000))
        if idle_event:
            emit("worker_idle", idleSince=idle_event[0], terminateAfter=idle_event[1])


def worker_loop() -> None:
    with STATE.lock:
        STATE.status = "idle"
        STATE.idle_since = time.time()
    capabilities = probe()
    with STATE.lock:
        STATE.capabilities = capabilities
    print(f"[GPU Render] FFmpeg flavor: {FFMPEG_FLAVOR}", flush=True)
    print(
        f"[GPU Render] capabilities: h264_nvenc={bool(capabilities.get('encoders', {}).get('h264Nvenc'))} "
        f"hevc_nvenc_main10={bool(capabilities.get('encoders', {}).get('hevcNvencMain10'))}",
        flush=True,
    )
    print(f"[GPU Render] required codec: {REQUIRED_CODEC or 'both'}", flush=True)
    emit_startup_events(capabilities)
    while True:
        job = STATE.work_queue.get()
        def run_job(next_job: dict[str, Any] = job) -> None:
            try:
                process_job(next_job)
            finally:
                STATE.work_queue.task_done()
        threading.Thread(target=run_job, name=f"gpu-render-{job['jobId'][:8]}", daemon=True).start()


def idle_watchdog() -> None:
    while True:
        time.sleep(1)
        with STATE.lock:
            should_expire = not STATE.draining and STATE.status == "idle" and STATE.terminate_after is not None and time.time() >= STATE.terminate_after
            if should_expire:
                STATE.draining = True
                STATE.status = "draining"
        if should_expire:
            emit("idle_expired")


class Handler(BaseHTTPRequestHandler):
    server_version = "SceneBuilderGpuRender/1.0"

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def authorized(self) -> bool:
        if self.headers.get("Authorization") != f"Bearer {POD_TOKEN}":
            self.send_json(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "Unauthorized"})
            return False
        return True

    def json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > 8 * 1024 * 1024:
            raise ValueError("Invalid request body length")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON object required")
        return value

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] in ("/", "/health"):
            with STATE.lock:
                self.send_json(HTTPStatus.OK, {"ok": True, "runtime": "scenebuilder-gpu-fast-render", "status": STATE.status, "activeJobIds": sorted(STATE.active_job_ids), "activeSlots": len(STATE.active_job_ids), "maxSlots": STATE.max_concurrent_jobs, "idle": STATE.status == "idle"})
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if not self.authorized():
            return
        try:
            payload = self.json_body()
        except Exception as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return
        if path == "/jobs":
            job_id = str(payload.get("jobId") or "").strip()
            clips = payload.get("clips")
            if not job_id or not isinstance(clips, list) or not clips:
                self.send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "jobId and non-empty clips are required"})
                return
            with STATE.lock:
                if STATE.draining:
                    self.send_json(HTTPStatus.CONFLICT, {"ok": False, "error": "Worker is draining"})
                    return
                if job_id in STATE.jobs:
                    self.send_json(HTTPStatus.OK, {"ok": True, "duplicate": True, "jobId": job_id})
                    return
                requested_slots = max(1, min(MAX_CONCURRENT_JOBS, int(payload.get("maxConcurrentJobs") or 1)))
                STATE.max_concurrent_jobs = requested_slots
                if len(STATE.active_job_ids) >= STATE.max_concurrent_jobs:
                    self.send_json(HTTPStatus.CONFLICT, {"ok": False, "error": f"Worker busy with {sorted(STATE.active_job_ids)}"})
                    return
                expected_vram_mb = max(0, int(payload.get("expectedPeakVramMb") or 0))
                if expected_vram_mb:
                    stats = gpu_stats()
                    total_vram_mb = int(stats.get("vramMb") or 0)
                    used_vram_mb = int(stats.get("vramUsedMb") or 0)
                    vram_budget_mb = int(total_vram_mb * 0.90)
                    if total_vram_mb and used_vram_mb + expected_vram_mb > vram_budget_mb:
                        self.send_json(HTTPStatus.CONFLICT, {
                            "ok": False,
                            "error": "VRAM_HEADROOM_EXHAUSTED",
                            "vramUsedMb": used_vram_mb,
                            "expectedPeakVramMb": expected_vram_mb,
                            "vramBudgetMb": vram_budget_mb,
                        })
                        return
                settings = dict(payload.get("settings") or {})
                settings["width"] = int(settings.get("width") or 1920)
                settings["height"] = int(settings.get("height") or 1080)
                settings["fps"] = int(settings.get("fps") or 30)
                physical_vcpus = max(1, os.cpu_count() or 1)
                cpu_budget = max(1, int((physical_vcpus * 0.90) // requested_slots))
                settings["_physicalVcpus"] = physical_vcpus
                settings["_cpuBudget"] = cpu_budget
                settings["_parallelClipWorkers"] = parallel_clip_worker_count(settings)
                settings["_ffmpegThreads"] = max(1, math.ceil(cpu_budget / settings["_parallelClipWorkers"]))
                payload["settings"] = settings
                payload["audioClips"] = list(payload.get("audioClips") or [])
                payload["jobId"] = job_id
                STATE.jobs[job_id] = {"cancelRequested": False}
                STATE.active_job_ids.add(job_id)
                STATE.status = "busy"
                STATE.idle_since = None
                STATE.terminate_after = None
                try:
                    STATE.work_queue.put_nowait(payload)
                except queue.Full:
                    STATE.active_job_ids.discard(job_id)
                    STATE.jobs.pop(job_id, None)
                    self.send_json(HTTPStatus.CONFLICT, {"ok": False, "error": "Worker queue is full"})
                    return
            self.send_json(HTTPStatus.ACCEPTED, {"ok": True, "accepted": True, "jobId": job_id})
            return
        if path.startswith("/jobs/") and path.endswith("/cancel"):
            job_id = path[len("/jobs/"):-len("/cancel")].strip("/")
            with STATE.lock:
                record = STATE.jobs.get(job_id)
                if not record:
                    self.send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Unknown job"})
                    return
                record["cancelRequested"] = True
                processes = list(STATE.active_processes.get(job_id, []))
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            self.send_json(HTTPStatus.ACCEPTED, {"ok": True, "cancelRequested": True, "jobId": job_id})
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[GPU Render HTTP] {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    if not POD_TOKEN or not WORKER_ID or not CONTROL_URL:
        raise RuntimeError("SCENEBUILDER_POD_TOKEN, SCENEBUILDER_WORKER_ID, and SCENEBUILDER_CONTROL_URL are required")
    threading.Thread(target=worker_loop, name="gpu-render-worker", daemon=True).start()
    threading.Thread(target=idle_watchdog, name="gpu-render-idle-watchdog", daemon=True).start()
    print(f"[GPU Render] listening on {HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()

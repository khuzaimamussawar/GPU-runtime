from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


HOST = os.environ.get("RENDER_GPU_HOST", "0.0.0.0")
PORT = int(os.environ.get("RENDER_GPU_PORT", "8000"))
POD_TOKEN = os.environ.get("SCENEBUILDER_POD_TOKEN", "").strip()
WORKER_ID = os.environ.get("SCENEBUILDER_WORKER_ID", "").strip()
CONTROL_URL = os.environ.get("SCENEBUILDER_CONTROL_URL", "").strip()
DEFAULT_IDLE_TIMEOUT = max(0, int(os.environ.get("SCENEBUILDER_IDLE_TIMEOUT_SECONDS", "60")))
REQUIRED_CODEC = os.environ.get("SCENEBUILDER_REQUIRED_CODEC", "").strip().lower()
MAX_ERROR_DETAIL_CHARS = 16000


class RenderError(RuntimeError):
    pass


class State:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.work_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self.active_job_id: str | None = None
        self.active_processes: list[subprocess.Popen[bytes]] = []
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


def disk_stats() -> dict[str, float]:
    usage = shutil.disk_usage("/")
    return {"diskUsedMb": round((usage.total - usage.free) / 1048576, 2), "diskFreeMb": round(usage.free / 1048576, 2)}


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
        "nvdec": "cuda" in command_output(["ffmpeg", "-hide_banner", "-hwaccels"]),
        "cudaFilters": "scale_cuda" in command_output(["ffmpeg", "-hide_banner", "-filters"]),
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


def ffmpeg_video_decoder_args(clip: dict[str, Any], settings: dict[str, Any]) -> list[str]:
    duration = max(0.001, float(clip.get("sceneDuration") or 0.001))
    media_type = str(clip.get("type") or "placeholder")
    source = str(clip.get("url") or "")
    if media_type == "placeholder" or not source:
        return ["-f", "lavfi", "-i", f"color=c=black:s={settings['width']}x{settings['height']}:r={settings['fps']}", "-t", str(duration)]
    if media_type == "image":
        return ["-loop", "1", "-framerate", str(settings["fps"]), "-i", source, "-t", str(duration)]
    speed = max(0.1, float(clip.get("speed") or 1))
    source_duration = duration * speed
    start = max(0, float(clip.get("startTimeOffset") or 0))
    return ["-accurate_seek", "-ss", str(start), "-t", str(source_duration), "-i", source]


def video_filter(clip: dict[str, Any], settings: dict[str, Any]) -> str:
    duration = max(0.001, float(clip.get("sceneDuration") or 0.001))
    speed = max(0.1, float(clip.get("speed") or 1))
    width, height, fps = settings["width"], settings["height"], settings["fps"]
    filters = [
        f"scale={width}:{height}:force_original_aspect_ratio=decrease",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
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
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo", "-t", str(duration), "-c:a", "pcm_s16le", str(output_path)], "audio silence")
        return
    base = float(clip.get("audioSourceStart") or 0) + float(clip.get("startTimeOffset") or 0)
    ranges = [item for item in (clip.get("keepRanges") or []) if float(item.get("end") or 0) > float(item.get("start") or 0)]
    filter_value = audio_filter(clip)
    if not ranges:
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(max(0, base)), "-i", source, "-t", str(duration), "-vn", "-af", f"{filter_value},apad", "-t", str(duration), "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(output_path)], "audio unit")
        return
    pieces: list[Path] = []
    for index, item in enumerate(ranges):
        start = float(item["start"])
        length = max(0.001, float(item["end"]) - start)
        piece = work_dir / f"audio-range-{output_path.stem}-{index}.wav"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(max(0, base + start)), "-i", source, "-t", str(length), "-vn", "-af", filter_value, "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(piece)], "retained audio range")
        pieces.append(piece)
    list_path = work_dir / f"{output_path.stem}.txt"
    list_path.write_text("\n".join(f"file '{path.as_posix()}'" for path in pieces), encoding="utf-8")
    joined = work_dir / f"{output_path.stem}-joined.wav"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c:a", "pcm_s16le", str(joined)], "join retained audio")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(joined), "-af", "apad", "-t", str(duration), "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(output_path)], "normalize retained audio")


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
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(output)], "canonical audio mix")
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
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(downloaded), "-af", "apad", "-t", str(total_seconds), "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(output)], "prepare project encoded audio")
    return output


def register_process(process: subprocess.Popen[bytes]) -> None:
    with STATE.lock:
        STATE.active_processes.append(process)


def unregister_process(process: subprocess.Popen[bytes]) -> None:
    with STATE.lock:
        if process in STATE.active_processes:
            STATE.active_processes.remove(process)


def raise_if_cancelled(job_id: str) -> None:
    with STATE.lock:
        if STATE.jobs.get(job_id, {}).get("cancelRequested"):
            raise RenderError("RENDER_CANCELLED")


def render_video(job: dict[str, Any], audio: Path | None, work_dir: Path) -> Path:
    settings = job["settings"]
    clips = job["clips"]
    output = work_dir / "video.mp4"
    codec = str(settings.get("codec") or "h264")
    encoder_args = ["-c:v", "hevc_nvenc", "-profile:v", "main10", "-pix_fmt", "p010le", "-preset", str(settings.get("preset") or "p6"), "-rc", "vbr", "-cq", str(settings.get("cq") or 17), "-tag:v", "hvc1"] if codec == "h265" else ["-c:v", "h264_nvenc", "-preset", str(settings.get("preset") or "p6"), "-rc", "vbr", "-cq", str(settings.get("cq") or 17), "-pix_fmt", "yuv420p"]
    encoder = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-video_size", f"{settings['width']}x{settings['height']}", "-framerate", str(settings["fps"]), "-i", "pipe:0", "-map", "0:v:0", "-an", *encoder_args, "-r", str(settings["fps"]), str(output)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    register_process(encoder)
    total = max(1, len(clips))
    try:
        for index, clip in enumerate(clips):
            raise_if_cancelled(job["jobId"])
            decoder = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *ffmpeg_video_decoder_args(clip, settings), "-map", "0:v:0", "-an", "-vf", video_filter(clip, settings), "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            register_process(decoder)
            assert decoder.stdout is not None and encoder.stdin is not None
            while chunk := decoder.stdout.read(1024 * 1024):
                raise_if_cancelled(job["jobId"])
                encoder.stdin.write(chunk)
            stderr = decoder.stderr.read().decode("utf-8", errors="replace") if decoder.stderr else ""
            if decoder.wait() != 0:
                raise RenderError(f"visual unit {index} failed: {stderr[-MAX_ERROR_DETAIL_CHARS:]}")
            unregister_process(decoder)
            progress = 20 + int(((index + 1) / total) * 70)
            emit("job_progress", jobId=job["jobId"], progress=progress, phase="video", clipIndex=index)
            emit("job_metric", jobId=job["jobId"], metric={"phase": "video", "clipIndex": index, "route": "streaming", "renderFps": 0, "resource": {**gpu_stats(), **disk_stats()}})
        assert encoder.stdin is not None
        encoder.stdin.close()
        stderr = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
        if encoder.wait() != 0:
            raise RenderError(f"final NVENC encode failed: {stderr[-MAX_ERROR_DETAIL_CHARS:]}")
    finally:
        unregister_process(encoder)
    if audio is None:
        return output
    muxed = work_dir / "output.mp4"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(output), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", str(settings.get("audioBitrate") or "384k"), "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(muxed)], "final audio mux")
    return muxed


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
            output = render_video(job, audio, work_dir)
            raise_if_cancelled(job_id)
            emit("job_progress", jobId=job_id, progress=95, phase="uploading")
            upload(str(job["uploadUrl"]), output)
            emit("job_done", jobId=job_id, outputUrl=job.get("outputUrl"), elapsedMs=int((time.time() - started) * 1000))
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
        with STATE.lock:
            STATE.active_job_id = None
            STATE.active_processes.clear()
            STATE.status = "idle"
            STATE.idle_since = time.time()
            timeout = int(job.get("idleTimeoutSeconds") or DEFAULT_IDLE_TIMEOUT)
            STATE.terminate_after = STATE.idle_since + timeout
        emit("worker_idle", idleSince=int(STATE.idle_since * 1000), terminateAfter=int(STATE.terminate_after * 1000))


def worker_loop() -> None:
    with STATE.lock:
        STATE.status = "idle"
        STATE.idle_since = time.time()
    capabilities = probe()
    print(f"[GPU Render] required codec: {REQUIRED_CODEC or 'both'}", flush=True)
    emit_startup_events(capabilities)
    while True:
        job = STATE.work_queue.get()
        try:
            process_job(job)
        finally:
            STATE.work_queue.task_done()


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
                self.send_json(HTTPStatus.OK, {"ok": True, "runtime": "scenebuilder-gpu-fast-render", "status": STATE.status, "activeJobId": STATE.active_job_id, "idle": STATE.status == "idle"})
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
                if STATE.active_job_id is not None or not STATE.work_queue.empty():
                    self.send_json(HTTPStatus.CONFLICT, {"ok": False, "error": f"Worker busy with {STATE.active_job_id}"})
                    return
                settings = dict(payload.get("settings") or {})
                settings["width"] = int(settings.get("width") or 1920)
                settings["height"] = int(settings.get("height") or 1080)
                settings["fps"] = int(settings.get("fps") or 30)
                payload["settings"] = settings
                payload["audioClips"] = list(payload.get("audioClips") or [])
                payload["jobId"] = job_id
                STATE.jobs[job_id] = {"cancelRequested": False}
                STATE.active_job_id = job_id
                STATE.status = "busy"
                STATE.idle_since = None
                STATE.terminate_after = None
                STATE.work_queue.put_nowait(payload)
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
                processes = list(STATE.active_processes)
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

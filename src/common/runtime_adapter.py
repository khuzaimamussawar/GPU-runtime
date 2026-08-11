from __future__ import annotations

import asyncio
import json
import math
import shutil
import subprocess
import time
import urllib.parse
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import src.common.h3_runtime as runtime
from src.common.job_contract import H3Job


# Keep the large, stable runtime module untouched while fixing production-specific
# workflow behavior here. The four provider handlers import run_h3_job from this
# adapter, so both RunPod and Novita execute the same H3 contract.
_ORIGINAL_PREPARE_WORKFLOW = runtime.prepare_workflow
_ORIGINAL_ENSURE_COMFY_RUNNING = runtime.ensure_comfy_running
_ORIGINAL_WAIT_FOR_OUTPUT_VIDEO = runtime.wait_for_output_video
_ORIGINAL_ENCODE_OUTPUTS = runtime.encode_outputs

# The shared SageAttention 2.2.0 artifact is intentionally compiled only for the
# production Ada + Blackwell architectures. Do not let a UI toggle force Sage on
# for an unsupported worker (for example an unknown Hopper-backed MIG slice).
_SAGE_COMPILED_CAPABILITIES = {(8, 9), (12, 0)}

ProgressCallback = Callable[[str, int | None], None]
_ACTIVE_PROGRESS_CALLBACK: ProgressCallback | None = None
_ACTIVE_JOB_ID: str | None = None


def _emit_progress(phase: str, progress: int | None = None) -> None:
    callback = _ACTIVE_PROGRESS_CALLBACK
    if callback is None:
        return
    try:
        callback(str(phase), None if progress is None else int(progress))
    except Exception as exc:
        # Progress reporting must never fail a generation job.
        print(f"[SceneBuilder H3] progress callback failed: {exc}")


def _remove_path(obj: dict[str, Any], dotted: str | None) -> None:
    if not dotted:
        return
    try:
        parent, key = runtime.resolve_parent(obj, dotted)
        if isinstance(parent, dict):
            parent.pop(key, None)
        elif isinstance(parent, list) and isinstance(key, int) and 0 <= key < len(parent):
            parent[key] = None
    except Exception:
        return


def _node_id_from_input_path(path: str | None) -> str | None:
    if not path:
        return None
    return str(path).split(".", 1)[0] or None


def _drop_node(workflow: dict[str, Any], node_id: str | None) -> None:
    if node_id:
        workflow.pop(str(node_id), None)


def _drop_literal_input(workflow: dict[str, Any], node_id: str, input_name: str) -> None:
    node = workflow.get(str(node_id))
    if not isinstance(node, dict):
        return
    inputs = node.get("inputs")
    if isinstance(inputs, dict):
        inputs.pop(input_name, None)


def _round_h3_frames(job: H3Job) -> int:
    """Round up to MiniMax H3's native 17*k+5 frame lattice."""
    requested = max(1, round(job.duration_seconds * job.fps))
    if requested <= 5:
        return 5
    if (requested - 5) % 17 == 0:
        return requested
    k = max(0, math.ceil((requested - 5) / 17))
    return 17 * k + 5


def _sage_was_requested(settings: dict[str, Any]) -> bool:
    value = settings.get("sageAttention")
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "off", "none", "disabled", "no"}


def _guard_sage_for_runtime_gpu(
    prepared: dict[str, Any], manifest: dict[str, Any], job: H3Job
) -> None:
    required = manifest.get("requiredPaths") or {}
    optional = manifest.get("optionalPaths") or {}
    sage_path = required.get("sageAttention") or optional.get("sageAttention")
    if not sage_path:
        return

    # Checkbox OFF always means the actual KJ node value `disabled`.
    if not _sage_was_requested(job.settings):
        runtime.set_path(prepared, sage_path, "disabled")
        return

    capability: tuple[int, int] | None = None
    try:
        # Import lazily so source-only CI/validation does not need a Torch install.
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            major, minor = torch.cuda.get_device_capability(0)
            capability = (int(major), int(minor))
    except Exception:
        capability = None

    if capability in _SAGE_COMPILED_CAPABILITIES:
        # Checkbox ON on a supported worker maps to KJ's automatic Sage choice.
        runtime.set_path(prepared, sage_path, "auto")
        return

    # Unsupported/unknown worker: disable Sage only. Never auto-enable Spectrum;
    # Spectrum remains exactly as requested in the independent spectrum settings.
    runtime.set_path(prepared, sage_path, "disabled")
    detected = "none" if capability is None else f"{capability[0]}.{capability[1]}"
    print(
        "[SceneBuilder H3] SageAttention requested but automatically disabled: "
        f"worker CUDA capability={detected}, compiled capabilities="
        f"{sorted(_SAGE_COMPILED_CAPABILITIES)}. Spectrum setting unchanged."
    )


def _normalize_preprocess_reference_fit(settings: dict[str, Any]) -> str:
    raw = str(
        settings.get("inputImageFit")
        or settings.get("sceneRefFit")
        or settings.get("referenceFit")
        or "auto"
    ).strip().lower().replace("_", " ")

    if raw in {
        "match", "crop", "fill", "frame", "match project frame",
        "match first image", "fill / crop", "fill/crop",
    }:
        return "match"
    if raw in {"contain", "pad", "fit", "fit full image", "fit full"}:
        return "contain"
    # Auto / max / unknown uses the less destructive max-size path for references.
    return "max"


def _normalize_native_reference_size(settings: dict[str, Any]) -> str:
    raw = str(settings.get("referenceFit") or settings.get("refImageSize") or "match")
    raw = raw.strip().lower().replace("_", " ")
    if raw in {"max", "fit full image", "fit full", "contain", "pad", "fit"}:
        return "max"
    return "match"


def _normalize_max_input_size(settings: dict[str, Any], job: H3Job) -> dict[str, Any]:
    normalized = dict(settings)
    key = "referenceMaxSize" if "referenceMaxSize" in normalized else "maxInputSize"
    if key not in normalized:
        return normalized

    raw = normalized.get(key)
    if isinstance(raw, (int, float)):
        normalized[key] = max(256, min(8192, int(raw)))
        return normalized

    label = str(raw or "auto").strip().lower().replace("_", " ")
    if label in {"original", "original / no downscale", "no downscale", "none"}:
        normalized[key] = "original"
    elif label in {"same as output", "output", "same"}:
        normalized[key] = max(job.width, job.height)
    elif label in {"720p", "720"}:
        normalized[key] = 1280
    elif label in {"1k", "1024", "1 k"}:
        normalized[key] = 1024
    elif label in {"1080p", "1080"}:
        normalized[key] = 1920
    else:
        normalized[key] = 1024
    return normalized


def _job_for_prepare(job: H3Job) -> H3Job:
    settings = _normalize_max_input_size(job.settings, job)
    # Keep preprocessing semantics and the native H3 combo separate:
    # inputImageFit drives our Pillow preprocessing while referenceFit is always
    # one of the native MiniMax H3 values (`match` or `max`). Ref2VA exposes one
    # ref_image_size for the whole job, so the user's selected value applies to
    # every image reference in that request.
    settings["inputImageFit"] = _normalize_preprocess_reference_fit(job.settings)
    settings["referenceFit"] = _normalize_native_reference_size(job.settings)
    return replace(job, settings=settings)


def _resize_inside_max(image: Any, settings: dict[str, Any]) -> Any:
    raw = settings.get("referenceMaxSize")
    if raw is None:
        raw = settings.get("maxInputSize")
    if isinstance(raw, str) and raw.strip().lower() == "original":
        return image
    try:
        max_size = int(raw or 1024)
    except (TypeError, ValueError):
        max_size = 1024
    max_size = max(256, min(8192, max_size))

    source_w, source_h = image.size
    longest = max(source_w, source_h)
    if longest <= max_size:
        return image
    scale = max_size / longest
    target = (max(32, round(source_w * scale)), max(32, round(source_h * scale)))
    return image.resize(target, runtime.resample_lanczos())


def _prepare_workflow(
    workflow: dict[str, Any],
    manifest: dict[str, Any],
    job: H3Job,
) -> dict[str, Any]:
    prepare_job = _job_for_prepare(job)
    prepared = _ORIGINAL_PREPARE_WORKFLOW(workflow, manifest, prepare_job)
    _guard_sage_for_runtime_gpu(prepared, manifest, job)
    output_path = (manifest.get("optionalPaths") or {}).get("outputPrefix")
    if output_path:
        runtime.set_path(
            prepared,
            output_path,
            f"video/scenebuilder/{runtime.safe_name(job.job_id)}/ComfyUI",
        )
    return prepared


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        # A media-ref object is one reference; an arbitrary keyed collection is
        # treated as a list of references.
        if any(key in value for key in ("url", "objectKey", "key")):
            return [value]
        return list(value.values())
    return [value]


def _prune_ref2va_placeholders(
    workflow: dict[str, Any],
    optional: dict[str, str],
    *,
    image_count: int,
    video_count: int,
    audio_count: int,
) -> None:
    # Comfy API serializes dynamic MiniMax H3 reference inputs as literal keys
    # containing dots (for example `ref_images.ref_image_0`). They are not nested
    # JSON objects, so generic dotted-path helpers cannot remove them safely.
    reference_node = str(optional.get("referenceNode") or "1")

    for index in range(image_count, 9):
        _drop_literal_input(workflow, reference_node, f"ref_images.ref_image_{index}")
        _drop_node(workflow, _node_id_from_input_path(optional.get(f"referenceImage{index}")))

    for index in range(video_count, 3):
        _drop_literal_input(workflow, reference_node, f"ref_videos.ref_video_{index}")
        _drop_literal_input(workflow, reference_node, f"ref_video_audios.ref_video_audio_{index}")
        _drop_node(workflow, _node_id_from_input_path(optional.get(f"referenceVideo{index}")))
        _drop_node(workflow, optional.get(f"referenceVideoComponent{index}"))

    for index in range(audio_count, 3):
        _drop_literal_input(workflow, reference_node, f"ref_audios.ref_audio_{index}")
        _drop_node(workflow, _node_id_from_input_path(optional.get(f"referenceAudio{index}")))


def _parse_ms(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(0, parsed)


def _media_ranges(ref: dict[str, Any]) -> list[tuple[int, int | None]]:
    start = _parse_ms(
        ref.get("sourceStartMs")
        if "sourceStartMs" in ref
        else ref.get("startMs", ref.get("trimStartMs"))
    )
    end = _parse_ms(
        ref.get("sourceEndMs")
        if "sourceEndMs" in ref
        else ref.get("endMs", ref.get("trimEndMs"))
    )
    if start is not None or end is not None:
        start = start or 0
        if end is None or end > start:
            return [(start, end)]

    ranges: list[tuple[int, int | None]] = []
    keep_ranges = ref.get("keepRanges") or ref.get("ranges") or []
    if isinstance(keep_ranges, dict):
        keep_ranges = list(keep_ranges.values())
    if not isinstance(keep_ranges, (list, tuple)):
        return ranges
    for item in keep_ranges:
        if isinstance(item, dict):
            item_start = _parse_ms(
                item.get("sourceStartMs", item.get("startMs", item.get("start")))
            )
            item_end = _parse_ms(
                item.get("sourceEndMs", item.get("endMs", item.get("end")))
            )
        elif isinstance(item, (list, tuple)) and item:
            item_start = _parse_ms(item[0])
            item_end = _parse_ms(item[1]) if len(item) > 1 else None
        else:
            continue
        item_start = item_start or 0
        if item_end is None or item_end > item_start:
            ranges.append((item_start, item_end))
    return ranges


def _absolute_input_path(relative: str) -> Path:
    return runtime.COMFY_INPUT / relative


def _relative_input_path(path: Path) -> str:
    return str(path.relative_to(runtime.COMFY_INPUT)).replace("\\", "/")


def _materialize_reference_video(
    ref: dict[str, Any] | None, target_dir: Path, index: int
) -> str:
    relative = runtime.download_media(ref, target_dir)
    source = _absolute_input_path(relative)
    output = target_dir / f"{source.stem}_h3_ref{index}_24fps.mp4"
    if output.exists():
        return _relative_input_path(output)

    cmd = ["ffmpeg", "-y"]
    ranges = _media_ranges(ref or {})
    if ranges:
        start_ms, end_ms = ranges[0]
        cmd += ["-ss", f"{start_ms / 1000:.6f}"]
        if end_ms is not None:
            cmd += ["-to", f"{end_ms / 1000:.6f}"]
    cmd += [
        "-i", str(source),
        "-map", "0:v:0",
        "-map", "0:a?",
        "-vf", "fps=24",
        "-fps_mode", "cfr",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    return _relative_input_path(output)


def _materialize_reference_audio(
    ref: dict[str, Any] | None, target_dir: Path, index: int
) -> str:
    relative = runtime.download_media(ref, target_dir)
    source = _absolute_input_path(relative)
    ranges = _media_ranges(ref or {})
    if not ranges:
        return relative

    output = target_dir / f"{source.stem}_h3_ref{index}_range.wav"
    if output.exists():
        return _relative_input_path(output)

    if len(ranges) == 1:
        start_ms, end_ms = ranges[0]
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_ms / 1000:.6f}",
        ]
        if end_ms is not None:
            cmd += ["-to", f"{end_ms / 1000:.6f}"]
        cmd += [
            "-i", str(source),
            "-vn",
            "-ac", "2",
            "-ar", "48000",
            "-c:a", "pcm_s16le",
            str(output),
        ]
    else:
        filter_parts: list[str] = []
        concat_inputs: list[str] = []
        for range_index, (start_ms, end_ms) in enumerate(ranges):
            end_expr = "" if end_ms is None else f":end={end_ms / 1000:.6f}"
            label = f"a{range_index}"
            filter_parts.append(
                f"[0:a]atrim=start={start_ms / 1000:.6f}{end_expr},"
                f"asetpts=PTS-STARTPTS[{label}]"
            )
            concat_inputs.append(f"[{label}]")
        filter_parts.append(
            "".join(concat_inputs) + f"concat=n={len(ranges)}:v=0:a=1[outa]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(source),
            "-filter_complex", ";".join(filter_parts),
            "-map", "[outa]",
            "-ac", "2",
            "-ar", "48000",
            "-c:a", "pcm_s16le",
            str(output),
        ]

    subprocess.run(cmd, check=True)
    return _relative_input_path(output)


def _materialize_reference_audio_segments(
    refs: list[Any], target_dir: Path
) -> str:
    """Materialize exact ordered Director source ranges into one H3 audio ref."""
    paths: list[Path] = []
    for index, value in enumerate(refs):
        ref = runtime.media_ref(value)
        if not ref:
            continue
        relative = _materialize_reference_audio(ref, target_dir, index)
        paths.append(_absolute_input_path(relative))

    if not paths:
        raise runtime.H3RuntimeError("referenceAudioSegments did not contain usable audio")
    if len(paths) == 1:
        return _relative_input_path(paths[0])

    output = target_dir / "director_audio_segments_h3.wav"
    if output.exists():
        return _relative_input_path(output)

    cmd = ["ffmpeg", "-y"]
    for source in paths:
        cmd += ["-i", str(source)]
    concat_inputs = "".join(f"[{index}:a:0]" for index in range(len(paths)))
    cmd += [
        "-filter_complex",
        f"{concat_inputs}concat=n={len(paths)}:v=0:a=1[outa]",
        "-map", "[outa]",
        "-ac", "2",
        "-ar", "48000",
        "-c:a", "pcm_s16le",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    return _relative_input_path(output)


def _materialize_inputs(
    workflow: dict[str, Any],
    required: dict[str, str],
    optional: dict[str, str],
    job: H3Job,
) -> None:
    inputs = job.inputs
    media_dir = runtime.COMFY_INPUT / "scenebuilder" / runtime.safe_name(job.job_id)
    media_dir.mkdir(parents=True, exist_ok=True)

    first_frame = runtime.media_ref(
        inputs.get("firstFrame") or inputs.get("startFrame") or inputs.get("image")
    )
    last_frame = runtime.media_ref(inputs.get("lastFrame") or inputs.get("endFrame"))

    if job.task_family == "h3_fl2va":
        if job.mode in {"i2v", "fflf"} and not first_frame:
            raise runtime.H3RuntimeError(f"{job.mode} requires firstFrame/startFrame/image input")
        if job.mode in {"fflf", "l2v"} and not last_frame:
            raise runtime.H3RuntimeError(f"{job.mode} requires lastFrame/endFrame input")

        if first_frame:
            runtime.set_path(
                workflow,
                required["firstFrameImage"],
                runtime.materialize_image(first_frame, media_dir, job, "match"),
            )
        else:
            _remove_path(workflow, optional.get("firstFrameBinding"))
            _drop_node(workflow, _node_id_from_input_path(required.get("firstFrameImage")))

        if last_frame:
            runtime.set_path(
                workflow,
                required["lastFrameImage"],
                runtime.materialize_image(last_frame, media_dir, job, "match"),
            )
        else:
            _remove_path(workflow, optional.get("lastFrameBinding"))
            _drop_node(workflow, _node_id_from_input_path(required.get("lastFrameImage")))
        return

    if job.task_family != "h3_ref2va":
        raise runtime.H3RuntimeError(f"Unsupported taskFamily: {job.task_family}")

    refs = _as_list(inputs.get("referenceImages") or inputs.get("references"))
    if not refs and first_frame:
        refs = [first_frame]

    video_refs = _as_list(inputs.get("referenceVideos") or inputs.get("referenceVideo") or inputs.get("video"))
    audio_segments = _as_list(inputs.get("referenceAudioSegments"))
    audio_refs = _as_list(inputs.get("referenceAudios") or inputs.get("referenceAudio") or inputs.get("audio"))

    if len(refs) > 9:
        raise runtime.H3RuntimeError("h3_ref2va supports at most 9 image references per job")
    if len(video_refs) > 3:
        raise runtime.H3RuntimeError("h3_ref2va supports at most 3 reference videos per job")
    if len(audio_refs) > 3:
        raise runtime.H3RuntimeError("h3_ref2va supports at most 3 standalone audio references per job")
    if not refs and not video_refs and not audio_refs and not audio_segments:
        raise runtime.H3RuntimeError(
            "h3_ref2va requires at least one image, video, or audio reference"
        )

    reference_fit = runtime.normalize_reference_fit(job.settings)

    for index, ref in enumerate(refs):
        loader_path = optional.get(f"referenceImage{index}")
        if not loader_path:
            raise runtime.H3RuntimeError(f"Missing workflow path for reference image {index}")
        runtime.set_path(
            workflow,
            loader_path,
            runtime.materialize_image(runtime.media_ref(ref), media_dir, job, reference_fit),
        )

    for index, ref in enumerate(video_refs):
        loader_path = optional.get(f"referenceVideo{index}")
        if not loader_path:
            raise runtime.H3RuntimeError(f"Missing workflow path for reference video {index}")
        runtime.set_path(
            workflow,
            loader_path,
            _materialize_reference_video(runtime.media_ref(ref), media_dir, index),
        )

    audio_binding_count = 0
    if audio_segments:
        loader_path = optional.get("referenceAudio0")
        if not loader_path:
            raise runtime.H3RuntimeError("Missing workflow path for Director reference audio")
        runtime.set_path(
            workflow,
            loader_path,
            _materialize_reference_audio_segments(audio_segments, media_dir),
        )
        audio_binding_count = 1
    else:
        for index, ref in enumerate(audio_refs):
            loader_path = optional.get(f"referenceAudio{index}")
            if not loader_path:
                raise runtime.H3RuntimeError(f"Missing workflow path for reference audio {index}")
            runtime.set_path(
                workflow,
                loader_path,
                _materialize_reference_audio(runtime.media_ref(ref), media_dir, index),
            )
        audio_binding_count = len(audio_refs)

    _prune_ref2va_placeholders(
        workflow,
        optional,
        image_count=len(refs),
        video_count=len(video_refs),
        audio_count=audio_binding_count,
    )


def _ensure_comfy_running() -> None:
    _emit_progress("preparing_model")
    _ORIGINAL_ENSURE_COMFY_RUNNING()


async def _wait_for_output_video_with_progress(
    prompt_id: str, started_at: float, client_id: str
) -> Path | None:
    try:
        import aiohttp
    except Exception:
        return None

    deadline = time.time() + int(runtime.os.environ.get("COMFY_JOB_TIMEOUT_SECONDS", "3600"))
    last_history: dict[str, Any] | None = None
    base = runtime.COMFY_URL.rstrip("/")
    ws_base = base.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    ws_url = f"{ws_base}/ws?clientId={urllib.parse.quote(client_id, safe='')}"

    timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_connect=10, sock_read=None)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                websocket = await session.ws_connect(ws_url, heartbeat=30)
            except Exception as exc:
                print(f"[SceneBuilder H3] Comfy progress websocket unavailable: {exc}")
                return None

            async with websocket:
                while time.time() < deadline:
                    try:
                        async with session.get(
                            f"{base}/history/{urllib.parse.quote(prompt_id, safe='')}",
                            timeout=aiohttp.ClientTimeout(total=20),
                        ) as response:
                            if response.ok:
                                last_history = await response.json()
                                video = runtime.find_video_in_history(last_history)
                                if video:
                                    return video
                    except Exception:
                        pass

                    try:
                        message = await websocket.receive(timeout=2.0)
                    except asyncio.TimeoutError:
                        continue
                    if message.type == aiohttp.WSMsgType.TEXT:
                        try:
                            payload = json.loads(message.data)
                        except Exception:
                            continue
                        data = payload.get("data") if isinstance(payload, dict) else None
                        if not isinstance(data, dict):
                            continue
                        event_prompt_id = str(data.get("prompt_id") or data.get("promptId") or "")
                        if event_prompt_id and event_prompt_id != prompt_id:
                            continue
                        if str(payload.get("type") or "") == "progress":
                            value = data.get("value")
                            maximum = data.get("max")
                            try:
                                value_num = float(value)
                                max_num = float(maximum)
                            except (TypeError, ValueError):
                                continue
                            if max_num > 0:
                                percent = max(0, min(100, round(value_num / max_num * 100)))
                                _emit_progress("generating", percent)
                    elif message.type in {
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    }:
                        break
    except Exception as exc:
        print(f"[SceneBuilder H3] Comfy progress listener failed: {exc}")
        return None

    scanned = runtime.newest_video_file(started_at)
    if scanned:
        return scanned
    if last_history:
        print(f"[SceneBuilder H3] Comfy progress listener ended without output: {last_history}")
    return None


def _wait_for_output_video(prompt_id: str, started_at: float) -> Path:
    _emit_progress("generating")
    client_id = _ACTIVE_JOB_ID
    if client_id:
        try:
            video = asyncio.run(_wait_for_output_video_with_progress(prompt_id, started_at, client_id))
            if video:
                return video
        except Exception as exc:
            print(f"[SceneBuilder H3] falling back to history-only Comfy polling: {exc}")
    return _ORIGINAL_WAIT_FOR_OUTPUT_VIDEO(prompt_id, started_at)


def _encode_outputs(source_video: Path, job: H3Job) -> dict[str, Path]:
    _emit_progress("encoding")
    return _ORIGINAL_ENCODE_OUTPUTS(source_video, job)


def _upload_outputs(paths: dict[str, Path], job: H3Job) -> dict[str, Any]:
    """Upload the two final files using the SceneBuilder plan's canonical keys."""
    _emit_progress("uploading")
    project = runtime.safe_name(job.project_id)
    job_name = runtime.safe_name(job.job_id)
    master_key = f"projects/{project}/video/generated/{job_name}-h265.mp4"
    preview_key = f"projects/{project}/video/previews/{job_name}-h264-preview.mp4"
    return {
        "master": runtime.upload_file(paths["master"], master_key, "video/mp4"),
        "preview": runtime.upload_file(paths["preview"], preview_key, "video/mp4"),
    }


def _cleanup_job_files(job_id: str, remove_outputs: bool) -> None:
    safe = runtime.safe_name(job_id)
    shutil.rmtree(runtime.COMFY_INPUT / "scenebuilder" / safe, ignore_errors=True)
    tmp_dir = Path("/tmp/scenebuilder-h3") / safe
    if remove_outputs:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(runtime.COMFY_OUTPUT / "video" / "scenebuilder" / safe, ignore_errors=True)


def _all_outputs_uploaded(result: dict[str, Any] | None) -> bool:
    if not result:
        return False
    outputs = result.get("outputs") or {}
    items = [item for item in outputs.values() if isinstance(item, dict)]
    return bool(items) and all(bool(item.get("uploaded")) for item in items)


def run_h3_job(
    payload: dict[str, Any],
    expected_task_family: str,
    runtime_name: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    global _ACTIVE_PROGRESS_CALLBACK, _ACTIVE_JOB_ID

    # Install deterministic process-wide patches once. One H3 job runs per worker,
    # so the active progress callback is never shared by simultaneous GPU jobs.
    runtime.normalize_frame_count = _round_h3_frames
    runtime.materialize_inputs = _materialize_inputs
    runtime.prepare_workflow = _prepare_workflow
    runtime.resize_inside_max = _resize_inside_max
    runtime.ensure_comfy_running = _ensure_comfy_running
    runtime.wait_for_output_video = _wait_for_output_video
    runtime.encode_outputs = _encode_outputs
    runtime.upload_outputs = _upload_outputs

    normalized = runtime.unwrap_payload(payload)
    normalized.setdefault("taskFamily", expected_task_family)
    job_id = str(normalized.get("jobId") or normalized.get("job_id") or normalized.get("id") or "unknown")
    result: dict[str, Any] | None = None
    previous_callback = _ACTIVE_PROGRESS_CALLBACK
    previous_job_id = _ACTIVE_JOB_ID
    _ACTIVE_PROGRESS_CALLBACK = progress_callback
    _ACTIVE_JOB_ID = job_id
    try:
        result = runtime.run_h3_job(payload, expected_task_family, runtime_name)
        # The base runtime reports raw requested frames; return the actual H3 frame
        # count used by the patched workflow so D1/debug metadata stays truthful.
        try:
            result["frames"] = _round_h3_frames(runtime.normalize_job(normalized))
        except Exception:
            pass
        _emit_progress("completed", 100)
        return result
    finally:
        _ACTIVE_PROGRESS_CALLBACK = previous_callback
        _ACTIVE_JOB_ID = previous_job_id
        # Always remove downloaded inputs. Encoded/source outputs are removed only
        # after R2 upload succeeded, so a misconfigured storage endpoint still
        # leaves local files available for debugging instead of deleting results.
        _cleanup_job_files(job_id, remove_outputs=_all_outputs_uploaded(result))

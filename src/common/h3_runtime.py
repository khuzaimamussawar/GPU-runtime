from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from src.common.job_contract import H3Job, normalize_job


ROOT = Path(os.environ.get("SCENEBUILDER_H3_ROOT", "/opt/scenebuilder-h3"))
COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))
COMFY_INPUT = COMFY_ROOT / "input"
COMFY_OUTPUT = COMFY_ROOT / "output"
COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"


class H3RuntimeError(RuntimeError):
    pass


def run_h3_job(payload: dict[str, Any], expected_task_family: str, runtime_name: str) -> dict[str, Any]:
    normalized_payload = unwrap_payload(payload)
    normalized_payload.setdefault("taskFamily", expected_task_family)
    job = normalize_job(normalized_payload)
    if job.task_family != expected_task_family:
        raise H3RuntimeError(
            f"{runtime_name} expected taskFamily={expected_task_family}, got {job.task_family}"
        )

    workflow, manifest = load_workflow(job.task_family)
    patched = prepare_workflow(workflow, manifest, job)
    started_at = time.time()
    ensure_comfy_running()
    prompt_id = submit_comfy_prompt(patched, job.job_id)
    source_video = wait_for_output_video(prompt_id, started_at)
    encoded = encode_outputs(source_video, job)
    uploaded = upload_outputs(encoded, job)

    return {
        "ok": True,
        "runtime": runtime_name,
        "jobId": job.job_id,
        "projectId": job.project_id,
        "taskFamily": job.task_family,
        "mode": job.mode,
        "width": job.width,
        "height": job.height,
        "fps": job.fps,
        "frames": job.frame_count,
        "durationSeconds": job.duration_seconds,
        "outputs": uploaded,
    }


def unwrap_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "input" in payload and isinstance(payload["input"], dict):
        return dict(payload["input"])
    return dict(payload)


def load_workflow(task_family: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if task_family == "h3_fl2va":
        workflow_name = "fl2va_master.json"
        manifest_name = "fl2va_manifest.json"
    elif task_family == "h3_ref2va":
        workflow_name = "ref2va_master.json"
        manifest_name = "ref2va_manifest.json"
    else:
        raise H3RuntimeError(f"Unsupported taskFamily: {task_family}")

    workflow_path = ROOT / "workflows" / workflow_name
    manifest_path = ROOT / "workflows" / "manifests" / manifest_name
    return load_json(workflow_path), load_json(manifest_path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prepare_workflow(workflow: dict[str, Any], manifest: dict[str, Any], job: H3Job) -> dict[str, Any]:
    prepared = copy.deepcopy(workflow)
    required = manifest.get("requiredPaths", {})
    optional = manifest.get("optionalPaths", {})
    model_files = manifest.get("modelFiles", {})
    settings = job.settings

    if job.mode not in set(manifest.get("modes", [])):
        raise H3RuntimeError(f"Mode {job.mode} is not supported by {manifest.get('taskFamily')}")

    set_path(prepared, required["prompt"], job.prompt)
    set_path(prepared, required["width"], job.width)
    set_path(prepared, required["height"], job.height)
    set_path(prepared, required["lengthFrames"], normalize_frame_count(job))
    set_path(prepared, required["diffusionModel"], model_files["diffusion"])
    set_path(prepared, required["clipName"], select_clip_name(model_files, settings))
    set_path(prepared, required["videoVae"], model_files["videoVae"])
    if "audioVae" in required and model_files.get("audioVae"):
        set_path(prepared, required["audioVae"], model_files["audioVae"])

    apply_basic_settings(prepared, required, optional, settings)
    materialize_inputs(prepared, required, optional, job)
    reject_unsupported_settings(manifest, settings)
    return prepared


def normalize_frame_count(job: H3Job) -> int:
    frames = max(1, round(job.duration_seconds * job.fps))
    remainder = frames % 24
    if remainder:
        frames += 24 - remainder
    return frames


def select_clip_name(model_files: dict[str, Any], settings: dict[str, Any]) -> str:
    choice = str(settings.get("textEncoder") or settings.get("qwenEncoder") or "nvfp4").lower()
    return model_files.get("clipOptions", {}).get(choice) or model_files["clipDefault"]


def apply_basic_settings(
    workflow: dict[str, Any],
    required: dict[str, str],
    optional: dict[str, str],
    settings: dict[str, Any],
) -> None:
    simple_paths = {
        "steps": required.get("steps"),
        "scheduler": required.get("scheduler"),
        "sampler": required.get("sampler"),
        "seed": required.get("seed") or optional.get("seed"),
        "sageAttention": required.get("sageAttention") or optional.get("sageAttention"),
        "referenceFit": required.get("referenceFit") or optional.get("referenceFit"),
    }
    for key, path in simple_paths.items():
        if path and key in settings:
            set_path(workflow, path, settings[key])

    if "spectrumEnabled" in required:
        spectrum = settings.get("spectrum") or {}
        enabled = bool(settings.get("spectrumEnabled", spectrum.get("enabled", False)))
        set_path(workflow, required["spectrumEnabled"], enabled)

    spectrum_path = optional.get("spectrum")
    spectrum_settings = settings.get("spectrum") or {}
    if spectrum_path and isinstance(spectrum_settings, dict):
        for key, value in spectrum_settings.items():
            dotted = f"{spectrum_path}.{key}"
            if path_exists(workflow, dotted):
                set_path(workflow, dotted, value)


def materialize_inputs(
    workflow: dict[str, Any],
    required: dict[str, str],
    optional: dict[str, str],
    job: H3Job,
) -> None:
    inputs = job.inputs
    media_dir = COMFY_INPUT / "scenebuilder" / safe_name(job.job_id)
    media_dir.mkdir(parents=True, exist_ok=True)

    first_frame = media_ref(inputs.get("firstFrame") or inputs.get("startFrame") or inputs.get("image"))
    last_frame = media_ref(inputs.get("lastFrame") or inputs.get("endFrame"))

    if job.mode in {"i2v", "fflf"} and not first_frame:
        raise H3RuntimeError(f"{job.mode} requires firstFrame/startFrame/image input")
    if job.mode == "fflf" and not last_frame:
        raise H3RuntimeError("fflf requires lastFrame/endFrame input")

    if first_frame and "firstFrameImage" in required:
        set_path(workflow, required["firstFrameImage"], materialize_image(first_frame, media_dir, job, "match"))
    if last_frame and "lastFrameImage" in required:
        set_path(workflow, required["lastFrameImage"], materialize_image(last_frame, media_dir, job, "match"))

    refs = inputs.get("referenceImages") or inputs.get("references") or []
    if isinstance(refs, dict):
        refs = list(refs.values())
    if job.task_family == "h3_ref2va" and not refs and not first_frame:
        raise H3RuntimeError("h3_ref2va requires at least one referenceImages entry or firstFrame")
    reference_fit = normalize_reference_fit(job.settings)
    for index, ref in enumerate(refs[:9]):
        path = optional.get(f"referenceImage{index}")
        if path:
            set_path(workflow, path, materialize_image(media_ref(ref), media_dir, job, reference_fit))

    if "firstReferenceImage" in required and not refs and first_frame:
        set_path(workflow, required["firstReferenceImage"], materialize_image(first_frame, media_dir, job, reference_fit))

    audio_ref = media_ref(inputs.get("referenceAudio") or inputs.get("audio"))
    if audio_ref and "firstReferenceAudio" in required:
        set_path(workflow, required["firstReferenceAudio"], download_media(audio_ref, media_dir))

    video_ref = media_ref(inputs.get("referenceVideo") or inputs.get("video"))
    if video_ref and "firstReferenceVideo" in required:
        set_path(workflow, required["firstReferenceVideo"], download_media(video_ref, media_dir))


def media_ref(value: Any) -> dict[str, Any] | None:
    if not value:
        return None
    if isinstance(value, str):
        if value.startswith("http://") or value.startswith("https://"):
            return {"url": value}
        return {"objectKey": value}
    if isinstance(value, dict):
        return value
    raise H3RuntimeError(f"Unsupported media reference: {type(value).__name__}")


def download_media(ref: dict[str, Any] | None, target_dir: Path) -> str:
    if not ref:
        raise H3RuntimeError("Missing required media reference")
    url = ref.get("url")
    object_key = ref.get("objectKey") or ref.get("key")
    if not url and object_key:
        public_url = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")
        if not public_url:
            raise H3RuntimeError("objectKey media requires R2_PUBLIC_URL in the runtime")
        url = f"{public_url}/{str(object_key).lstrip('/')}"
    if not url:
        raise H3RuntimeError("Media reference must include url or objectKey")

    parsed = urllib.parse.urlparse(url)
    suffix = Path(parsed.path).suffix or ".bin"
    base = safe_name(Path(parsed.path).name or str(uuid.uuid4()))
    filename = base if base.endswith(suffix) else f"{base}{suffix}"
    output = target_dir / filename
    if not output.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            output.write_bytes(response.read())
    return str(output.relative_to(COMFY_INPUT)).replace("\\", "/")


def materialize_image(ref: dict[str, Any] | None, target_dir: Path, job: H3Job, fit: str) -> str:
    relative = download_media(ref, target_dir)
    source = COMFY_INPUT / relative
    if source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        return relative
    prepared = preprocess_image_for_h3(source, job, fit)
    return str(prepared.relative_to(COMFY_INPUT)).replace("\\", "/")


def normalize_reference_fit(settings: dict[str, Any]) -> str:
    value = str(
        settings.get("inputImageFit")
        or settings.get("sceneRefFit")
        or settings.get("referenceFit")
        or "max"
    ).lower()
    if value in {"match", "crop", "fill", "frame"}:
        return "match"
    if value in {"contain", "pad", "fit"}:
        return "contain"
    return "max"


def preprocess_image_for_h3(source: Path, job: H3Job, fit: str) -> Path:
    try:
        from PIL import Image, ImageOps
    except Exception as exc:
        raise H3RuntimeError("H3 image preprocessing requires Pillow in the runtime image") from exc

    target_w, target_h = job.width, job.height
    output = source.with_name(f"{source.stem}_h3_{fit}_{target_w}x{target_h}.png")
    if output.exists():
        return output

    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")

    if fit == "match":
        processed = resize_to_exact_aspect(image, target_w, target_h)
    elif fit == "contain":
        processed = contain_to_exact_aspect(image, target_w, target_h)
    else:
        processed = resize_inside_max(image, job.settings)

    processed.save(output, "PNG", optimize=True)
    return output


def resize_to_exact_aspect(image: Any, target_w: int, target_h: int) -> Any:
    target_ratio = target_w / target_h
    source_w, source_h = image.size
    source_ratio = source_w / source_h
    if source_ratio > target_ratio:
        crop_w = round(source_h * target_ratio)
        left = max(0, (source_w - crop_w) // 2)
        box = (left, 0, left + crop_w, source_h)
    else:
        crop_h = round(source_w / target_ratio)
        top = max(0, (source_h - crop_h) // 2)
        box = (0, top, source_w, top + crop_h)
    return image.crop(box).resize((target_w, target_h), resample_lanczos())


def contain_to_exact_aspect(image: Any, target_w: int, target_h: int) -> Any:
    from PIL import ImageOps

    return ImageOps.pad(image, (target_w, target_h), method=resample_lanczos(), color=(0, 0, 0), centering=(0.5, 0.5))


def resize_inside_max(image: Any, settings: dict[str, Any]) -> Any:
    source_w, source_h = image.size
    max_size = int(settings.get("referenceMaxSize") or settings.get("maxInputSize") or 1024)
    max_size = max(256, min(2048, max_size))
    longest = max(source_w, source_h)
    if longest <= max_size:
        return image
    scale = max_size / longest
    target = (max(32, round(source_w * scale)), max(32, round(source_h * scale)))
    return image.resize(target, resample_lanczos())


def resample_lanczos() -> Any:
    from PIL import Image

    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def reject_unsupported_settings(manifest: dict[str, Any], settings: dict[str, Any]) -> None:
    supported = {
        "textEncoder",
        "qwenEncoder",
        "steps",
        "scheduler",
        "sampler",
        "seed",
        "sageAttention",
        "referenceFit",
        "inputImageFit",
        "sceneRefFit",
        "referenceMaxSize",
        "maxInputSize",
        "spectrum",
        "spectrumEnabled",
    }
    unsupported = sorted(key for key in settings if key not in supported)
    if unsupported:
        raise H3RuntimeError(
            "Unsupported H3 setting(s) for this baked workflow: " + ", ".join(unsupported)
        )


def ensure_comfy_running() -> None:
    if comfy_is_ready():
        return
    cmd = [
        "python3",
        "main.py",
        "--listen",
        COMFY_HOST,
        "--port",
        str(COMFY_PORT),
        "--disable-auto-launch",
    ]
    subprocess.Popen(cmd, cwd=COMFY_ROOT)
    deadline = time.time() + int(os.environ.get("COMFY_STARTUP_TIMEOUT_SECONDS", "180"))
    while time.time() < deadline:
        if comfy_is_ready():
            return
        time.sleep(2)
    raise H3RuntimeError("ComfyUI did not become ready before timeout")


def comfy_is_ready() -> bool:
    try:
        request_json("GET", f"{COMFY_URL}/system_stats", None, timeout=3)
        return True
    except Exception:
        return False


def submit_comfy_prompt(workflow: dict[str, Any], client_id: str) -> str:
    response = request_json(
        "POST",
        f"{COMFY_URL}/prompt",
        {"prompt": workflow, "client_id": client_id},
        timeout=30,
    )
    prompt_id = response.get("prompt_id")
    if not prompt_id:
        raise H3RuntimeError(f"ComfyUI did not return prompt_id: {response}")
    return str(prompt_id)


def wait_for_output_video(prompt_id: str, started_at: float) -> Path:
    deadline = time.time() + int(os.environ.get("COMFY_JOB_TIMEOUT_SECONDS", "3600"))
    last_history: dict[str, Any] | None = None
    while time.time() < deadline:
        history = request_json("GET", f"{COMFY_URL}/history/{prompt_id}", None, timeout=20)
        last_history = history
        video = find_video_in_history(history)
        if video:
            return video
        time.sleep(3)

    scanned = newest_video_file(started_at)
    if scanned:
        return scanned
    raise H3RuntimeError(f"No ComfyUI video output found for {prompt_id}: {last_history}")


def find_video_in_history(history: dict[str, Any]) -> Path | None:
    video_exts = {".mp4", ".mov", ".webm", ".mkv"}
    stack: list[Any] = [history]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            filename = item.get("filename")
            if isinstance(filename, str) and Path(filename).suffix.lower() in video_exts:
                subfolder = str(item.get("subfolder") or "")
                file_type = str(item.get("type") or "output")
                base = COMFY_OUTPUT if file_type == "output" else COMFY_INPUT
                candidate = base / subfolder / filename
                if candidate.exists():
                    return candidate
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return None


def newest_video_file(started_at: float) -> Path | None:
    candidates = []
    for path in COMFY_OUTPUT.rglob("*"):
        if path.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"} and path.stat().st_mtime >= started_at:
            candidates.append(path)
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def encode_outputs(source_video: Path, job: H3Job) -> dict[str, Path]:
    output_dir = Path("/tmp/scenebuilder-h3") / safe_name(job.job_id)
    master = output_dir / f"{safe_name(job.job_id)}_h265.mp4"
    preview = output_dir / f"{safe_name(job.job_id)}_h264_preview.mp4"
    preview_scale = "854:-2" if job.width >= job.height else "480:-2"
    subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "encode_outputs.sh"),
            str(source_video),
            str(master),
            str(preview),
            preview_scale,
        ],
        check=True,
    )
    return {"master": master, "preview": preview}


def upload_outputs(paths: dict[str, Path], job: H3Job) -> dict[str, Any]:
    prefix = str(job.inputs.get("outputPrefix") or "").strip("/")
    if not prefix:
        prefix = f"projects/{job.project_id}/scene_videos"
    master_key = f"{prefix}/original/{safe_name(job.job_id)}_h265.mp4"
    preview_key = f"{prefix}/preview/{safe_name(job.job_id)}_h264_preview.mp4"

    outputs = {
        "master": upload_file(paths["master"], master_key, "video/mp4"),
        "preview": upload_file(paths["preview"], preview_key, "video/mp4"),
    }
    return outputs


def upload_file(path: Path, object_key: str, content_type: str) -> dict[str, Any]:
    bucket = os.environ.get("R2_BUCKET_NAME")
    endpoint = os.environ.get("R2_ENDPOINT")
    access_key = os.environ.get("R2_ACCESS_KEY")
    secret_key = os.environ.get("R2_SECRET_KEY")
    public_url = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")
    if not all([bucket, endpoint, access_key, secret_key]):
        return {"objectKey": object_key, "localPath": str(path), "uploaded": False}

    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=os.environ.get("R2_REGION", "auto"),
    )
    client.upload_file(
        str(path),
        bucket,
        object_key,
        ExtraArgs={"ContentType": content_type},
    )
    return {
        "objectKey": object_key,
        "url": f"{public_url}/{object_key}" if public_url else None,
        "uploaded": True,
        "sizeBytes": path.stat().st_size,
    }


def request_json(method: str, url: str, payload: dict[str, Any] | None, timeout: int) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise H3RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    return json.loads(body or "{}")


def set_path(obj: dict[str, Any], dotted: str, value: Any) -> None:
    parent, key = resolve_parent(obj, dotted)
    parent[key] = value


def path_exists(obj: dict[str, Any], dotted: str) -> bool:
    try:
        parent, key = resolve_parent(obj, dotted)
        if isinstance(parent, list):
            return isinstance(key, int) and 0 <= key < len(parent)
        return isinstance(parent, dict) and key in parent
    except Exception:
        return False


def resolve_parent(obj: dict[str, Any], dotted: str) -> tuple[Any, Any]:
    current: Any = obj
    parts = dotted.split(".")
    for part in parts[:-1]:
        current = current[int(part)] if isinstance(current, list) else current[part]
    last = parts[-1]
    return current, int(last) if isinstance(current, list) else last


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120] or "media"

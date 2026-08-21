from __future__ import annotations

import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import cv2

from .models import upscale_bgr
from .r2_store import upload_file
from .video_pipeline import run_fast_video

Progress = Callable[[str, float, dict[str, Any] | None], None]

IMAGE_MODELS = {
    "realesrgan-anime": "RealESRGAN_x4plus_anime_6B",
    "realesr-anime": "RealESRGAN_x4plus_anime_6B",
    "RealESRGAN_x4plus_anime_6B": "RealESRGAN_x4plus_anime_6B",
    "realesrgan-real": "RealESRGAN_x4plus",
    "realesr-real": "RealESRGAN_x4plus",
    "RealESRGAN_x4plus": "RealESRGAN_x4plus",
}


def _download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "SceneBuilder-Enhancer/2.0"})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as output:
        while True:
            chunk = response.read(4 * 1024 * 1024)
            if not chunk:
                break
            output.write(chunk)


def _target_long_side(target: str) -> int:
    value = str(target).lower()
    if value in {"4k", "2160p"}:
        return 3840
    if value in {"2k", "1440p"}:
        return 2560
    if value == "1080p":
        return 1920
    raise ValueError(f"Unsupported image target: {target}")


def _resolve_image_model(job: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    settings = job.get("settings") or {}
    requested = str(job.get("modelFamily") or settings.get("upscalerModel") or "realesrgan-real")
    model_name = IMAGE_MODELS.get(requested)
    if not model_name:
        raise ValueError(f"Unsupported Storyboard FAST image model: {requested}")
    return model_name, settings


def _upscale_one(source: str, output_key: str, model_name: str, settings: dict[str, Any], root: Path) -> dict[str, Any]:
    if not source.startswith(("http://", "https://")):
        raise ValueError("image_upscale input.url must be HTTP(S)")
    if not output_key:
        raise ValueError("output.objectKey is required")
    input_path = root / f"{abs(hash(source))}.input"
    final = root / f"{abs(hash(output_key))}.png"
    _download(source, input_path)
    frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("FFMPEG_DECODE_FAILED:image decode")
    target = _target_long_side(settings.get("targetResolution") or "2k")
    h, w = frame.shape[:2]
    scale = min(4.0, max(1.0, target / max(w, h)))
    enhanced = frame if scale <= 1.0 else upscale_bgr(frame, model_name, outscale=scale)
    h2, w2 = enhanced.shape[:2]
    if max(w2, h2) > target:
        ratio = target / max(w2, h2)
        enhanced = cv2.resize(enhanced, (max(1, round(w2 * ratio)), max(1, round(h2 * ratio))), interpolation=cv2.INTER_LANCZOS4)
    if not cv2.imwrite(str(final), enhanced, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError("IMAGE_ENCODE_FAILED")
    stored = upload_file(final, output_key, "image/png")
    return {**stored, "width": int(enhanced.shape[1]), "height": int(enhanced.shape[0])}


def run_image_upscale(job: dict[str, Any], cancel_event, progress: Progress) -> dict[str, Any]:
    source = str((job.get("input") or {}).get("url") or "").strip()
    output_key = str((job.get("output") or {}).get("objectKey") or "").strip()
    model_name, settings = _resolve_image_model(job)
    with tempfile.TemporaryDirectory(prefix="sb-enhancer-image-") as tmp:
        root = Path(tmp)
        progress("downloading", 5, None)
        progress("upscaling", 20, {"model": model_name, "precision": "fp16", "tile": 0})
        stored = _upscale_one(source, output_key, model_name, settings, root)
        if cancel_event.is_set():
            raise RuntimeError("CANCELLED")
        progress("uploading", 90, None)
        progress("completed", 100, None)
        return {
            **stored,
            "runtime": "scenebuilder-enhancer-fast",
            "modelFamily": model_name,
            "precision": "fp16",
            "targetResolution": settings.get("targetResolution") or "2k",
        }


def run_image_upscale_batch(job: dict[str, Any], cancel_event, progress: Progress) -> dict[str, Any]:
    model_name, settings = _resolve_image_model(job)
    images = (job.get("input") or {}).get("images") or []
    if not isinstance(images, list) or not images:
        raise ValueError("image_upscale_batch input.images is required")
    with tempfile.TemporaryDirectory(prefix="sb-enhancer-image-batch-") as tmp:
        root = Path(tmp)
        outputs: list[dict[str, Any] | None] = [None] * len(images)
        total = len(images)
        workers = max(1, min(int(settings.get("parallelism") or 1), int(settings.get("maxParallelism") or 4), total))

        def one(index: int, item: dict[str, Any]) -> dict[str, Any]:
            if cancel_event.is_set():
                raise RuntimeError("CANCELLED")
            source = str(item.get("url") or "").strip()
            output_key = str(item.get("objectKey") or "").strip()
            stored = _upscale_one(source, output_key, model_name, settings, root)
            return {**stored, "sceneId": item.get("sceneId"), "index": index}

        def run_sequential(start_count: int = 0) -> None:
            completed = start_count
            for index, item in enumerate(images):
                if outputs[index] is not None:
                    continue
                progress("upscaling", 5 + (completed / max(1, total)) * 85, {"model": model_name, "precision": "fp16", "tile": 0, "index": index + 1, "total": total, "xN": 1})
                outputs[index] = one(index, item)
                completed += 1

        if workers <= 1:
            run_sequential()
        else:
            completed = 0
            try:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(one, index, item): index for index, item in enumerate(images)}
                    for future in as_completed(futures):
                        index = futures[future]
                        outputs[index] = future.result()
                        completed += 1
                        progress("upscaling", 5 + (completed / max(1, total)) * 85, {"model": model_name, "precision": "fp16", "tile": 0, "index": index + 1, "total": total, "xN": workers})
            except Exception as error:
                if "out of memory" not in str(error).lower() and "CUDA_OOM" not in str(error):
                    raise
                if not settings.get("oomAutoBackoff", True):
                    raise
                progress("upscaling", 5 + (completed / max(1, total)) * 85, {"model": model_name, "oomBackoff": True, "xN": 1})
                run_sequential(completed)

        compact_outputs = [item for item in outputs if item is not None]
        progress("completed", 100, {"count": len(outputs)})
        return {
            "runtime": "scenebuilder-enhancer-fast",
            "modelFamily": model_name,
            "precision": "fp16",
            "targetResolution": settings.get("targetResolution") or "2k",
            "items": compact_outputs,
            "count": len(compact_outputs),
        }


__all__ = ["run_image_upscale", "run_image_upscale_batch", "run_fast_video"]

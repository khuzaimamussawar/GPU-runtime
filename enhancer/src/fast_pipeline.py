from __future__ import annotations

import tempfile
import urllib.request
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


def run_image_upscale(job: dict[str, Any], cancel_event, progress: Progress) -> dict[str, Any]:
    source = str((job.get("input") or {}).get("url") or "").strip()
    output_key = str((job.get("output") or {}).get("objectKey") or "").strip()
    settings = job.get("settings") or {}
    if not source.startswith(("http://", "https://")):
        raise ValueError("image_upscale input.url must be HTTP(S)")
    if not output_key:
        raise ValueError("output.objectKey is required")
    requested = str(job.get("modelFamily") or settings.get("upscalerModel") or "realesrgan-real")
    model_name = IMAGE_MODELS.get(requested)
    if not model_name:
        raise ValueError(f"Unsupported Storyboard FAST image model: {requested}")

    with tempfile.TemporaryDirectory(prefix="sb-enhancer-image-") as tmp:
        root = Path(tmp); input_path = root / "input"; final = root / "final.png"
        progress("downloading", 5, None); _download(source, input_path)
        if cancel_event.is_set(): raise RuntimeError("CANCELLED")
        frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("FFMPEG_DECODE_FAILED:image decode")
        target = _target_long_side(settings.get("targetResolution") or "2k")
        h, w = frame.shape[:2]
        scale = min(4.0, max(1.0, target / max(w, h)))
        progress("upscaling", 20, {"model": model_name, "precision": "fp16", "tile": 0})
        enhanced = frame if scale <= 1.0 else upscale_bgr(frame, model_name, outscale=scale)
        if cancel_event.is_set(): raise RuntimeError("CANCELLED")
        h2, w2 = enhanced.shape[:2]
        if max(w2, h2) > target:
            ratio = target / max(w2, h2)
            enhanced = cv2.resize(enhanced, (max(1, round(w2 * ratio)), max(1, round(h2 * ratio))), interpolation=cv2.INTER_LANCZOS4)
        if not cv2.imwrite(str(final), enhanced, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError("IMAGE_ENCODE_FAILED")
        progress("uploading", 90, None)
        stored = upload_file(final, output_key, "image/png")
        progress("completed", 100, None)
        return {
            **stored,
            "runtime": "scenebuilder-enhancer-fast",
            "modelFamily": model_name,
            "precision": "fp16",
            "targetResolution": settings.get("targetResolution") or "2k",
            "width": int(enhanced.shape[1]),
            "height": int(enhanced.shape[0]),
        }


__all__ = ["run_image_upscale", "run_fast_video"]

from __future__ import annotations

from typing import Any

TARGETS = {
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2k": (2560, 1440),
    "2160p": (3840, 2160),
    "4k": (3840, 2160),
}


def normalized_aspect_ratio(value: Any) -> str | None:
    raw = str(value or "").strip().replace(" ", "")
    if raw in {"16:9", "16/9", "1.7777777778"}:
        return "16:9"
    if raw in {"9:16", "9/16", "0.5625"}:
        return "9:16"
    return None


def target_dimensions(target: str, aspect_ratio: Any = None, *, fallback_width: int = 0, fallback_height: int = 0) -> tuple[int, int]:
    dimensions = TARGETS.get(str(target or "").lower())
    if not dimensions:
        raise ValueError(f"Unsupported target resolution: {target}")
    aspect = normalized_aspect_ratio(aspect_ratio)
    portrait = aspect == "9:16" or (aspect is None and fallback_height > fallback_width)
    return dimensions[::-1] if portrait else dimensions


def center_crop_to_aspect(frame: Any, target_width: int, target_height: int) -> Any:
    height, width = frame.shape[:2]
    source_ratio = width / height
    target_ratio = target_width / target_height
    if source_ratio > target_ratio:
        crop_width = max(1, round(height * target_ratio))
        left = max(0, (width - crop_width) // 2)
        frame = frame[:, left:left + crop_width]
    elif source_ratio < target_ratio:
        crop_height = max(1, round(width / target_ratio))
        top = max(0, (height - crop_height) // 2)
        frame = frame[top:top + crop_height, :]
    return frame


def center_crop_resize(frame: Any, target_width: int, target_height: int) -> Any:
    frame = center_crop_to_aspect(frame, target_width, target_height)
    if frame.shape[1] == target_width and frame.shape[0] == target_height:
        return frame
    import cv2
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)


def ffmpeg_center_crop_filter(target_width: int, target_height: int) -> str:
    return (
        f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={target_width}:{target_height}"
    )

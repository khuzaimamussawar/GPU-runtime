from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from math import floor, isclose
from typing import Iterable, Sequence

EPSILON = 1e-8


@dataclass(frozen=True)
class Sample:
    output_index: int
    output_time: float
    left_index: int
    right_index: int
    alpha: float
    interpolate: bool
    across_scene_cut: bool = False


def is_integer_multiplier(source_fps: float, target_fps: float, playback_speed: float = 1.0) -> int | None:
    effective_source = float(source_fps) * float(playback_speed)
    if effective_source <= 0 or target_fps <= 0:
        return None
    ratio = float(target_fps) / effective_source
    rounded = round(ratio)
    if rounded >= 1 and isclose(ratio, rounded, rel_tol=0.0, abs_tol=1e-8):
        return int(rounded)
    return None


def schedule_cfr(
    source_fps: float,
    target_fps: float,
    source_frame_count: int,
    *,
    playback_speed: float = 1.0,
    scene_cut_after: Iterable[int] = (),
) -> list[Sample]:
    """Map output timestamps to source frame pairs.

    When playback speed is timing-baked, source position is
    output_time * source_fps * playback_speed. For speed=.5, a 5 second source
    becomes a 10 second derivative. Hard scene cuts are never interpolated.
    """
    if source_fps <= 0 or target_fps <= 0 or source_frame_count <= 0 or playback_speed <= 0:
        raise ValueError("source_fps, target_fps, frame count and playback_speed must be positive")
    cuts = set(int(value) for value in scene_cut_after)
    source_duration = source_frame_count / source_fps
    output_duration = source_duration / playback_speed
    output_count = max(1, round(output_duration * target_fps))
    result: list[Sample] = []
    for output_index in range(output_count):
        output_time = output_index / target_fps
        position = output_time * source_fps * playback_speed
        left = min(source_frame_count - 1, max(0, floor(position + EPSILON)))
        alpha = position - floor(position + EPSILON)
        if abs(alpha) < EPSILON or left >= source_frame_count - 1:
            result.append(Sample(output_index, output_time, left, left, 0.0, False))
            continue
        right = left + 1
        if left in cuts:
            # Use the nearest authoritative endpoint; never synthesize a cross-cut morph.
            endpoint = left if alpha < 0.5 else right
            result.append(Sample(output_index, output_time, endpoint, endpoint, 0.0, False, True))
            continue
        result.append(Sample(output_index, output_time, left, right, max(0.0, min(1.0, alpha)), True))
    return result


def schedule_vfr(
    source_pts: Sequence[float],
    target_fps: float,
    *,
    playback_speed: float = 1.0,
    scene_cut_after: Iterable[int] = (),
) -> list[Sample]:
    """Schedule against authoritative source PTS for VFR media."""
    if len(source_pts) < 1 or target_fps <= 0 or playback_speed <= 0:
        raise ValueError("source_pts and positive target_fps/playback_speed are required")
    pts = [float(value) for value in source_pts]
    if any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError("source_pts must be strictly increasing")
    if len(pts) == 1:
        return [Sample(0, 0.0, 0, 0, 0.0, False)]
    nominal_tail = pts[-1] - pts[-2]
    source_end = pts[-1] + nominal_tail
    output_duration = source_end / playback_speed
    output_count = max(1, round(output_duration * target_fps))
    cuts = set(int(value) for value in scene_cut_after)
    result: list[Sample] = []
    for output_index in range(output_count):
        out_t = output_index / target_fps
        source_t = out_t * playback_speed
        right = bisect_right(pts, source_t)
        if right <= 0:
            result.append(Sample(output_index, out_t, 0, 0, 0.0, False))
            continue
        if right >= len(pts):
            idx = len(pts) - 1
            result.append(Sample(output_index, out_t, idx, idx, 0.0, False))
            continue
        left = right - 1
        if isclose(source_t, pts[left], rel_tol=0.0, abs_tol=EPSILON):
            result.append(Sample(output_index, out_t, left, left, 0.0, False))
            continue
        alpha = (source_t - pts[left]) / (pts[right] - pts[left])
        if left in cuts:
            endpoint = left if alpha < 0.5 else right
            result.append(Sample(output_index, out_t, endpoint, endpoint, 0.0, False, True))
        else:
            result.append(Sample(output_index, out_t, left, right, alpha, True))
    return result

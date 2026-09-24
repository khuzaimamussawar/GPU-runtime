from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PreparedVideo:
    path: Path
    parts: list[dict[str, Any]]
    source_breaks_ms: list[float]
    raw_source_breaks_ms: list[float]
    timeline_parts: list[dict[str, float]]
    timing_baked: bool

    def minimum_output_frame_count(self, fps: float) -> int:
        """Return the non-short final frame count for this Director selection.

        A Director endpoint is expressed in milliseconds and does not always
        land exactly on a CFR frame boundary.  Always round the cumulative
        endpoint up so every prepared part together covers the requested
        timeline duration; never round a fractional final frame down.
        """
        safe_fps = max(0.001, float(fps))
        output_end_ms = float(self.timeline_parts[-1]["outputEndMs"]) if self.timeline_parts else 0.0
        return max(1, math.ceil((output_end_ms * safe_fps / 1000.0) - 1e-9))


def probe_video_frame_count(path: Path) -> int:
    """Decode-count a completed video stream for the final timing invariant."""
    payload = subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames", "-of", "json", str(path),
    ], text=True)
    import json
    streams = json.loads(payload).get("streams") or []
    value = (streams[0] if streams else {}).get("nb_read_frames")
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _number(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
        return parsed if parsed >= 0 else fallback
    except (TypeError, ValueError):
        return fallback


def _atempo_chain(speed: float) -> str:
    """Return a legal ffmpeg atempo chain for one bounded source part."""
    remaining = speed
    stages: list[float] = []
    while remaining < 0.5:
        stages.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        stages.append(2.0)
        remaining /= 2.0
    stages.append(remaining)
    return ",".join(f"atempo={value:.8f}" for value in stages)


def prepare_exact_source(
    source: Path,
    destination: Path,
    raw_parts: Any,
    *,
    source_duration_ms: float,
    has_audio: bool,
    timing_baked: bool,
    speed: float,
) -> PreparedVideo:
    """Make the exact Director-selected source sequence and its output map.

    The Worker supplies integer-ms trim endpoints. ffmpeg's trim filters decode
    to those timestamps, then concat keeps only the requested ranges. FFV1/PCM
    make this a lossless temporary, never a second lossy deliverable encode.
    """
    raw = raw_parts if isinstance(raw_parts, list) else []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        start = min(source_duration_ms, _number(item.get("sourceTrimInMs", item.get("source_trim_in_ms")), 0.0))
        requested_end = _number(item.get("sourceTrimOutMs", item.get("source_trim_out_ms")), source_duration_ms)
        end = min(source_duration_ms, requested_end if requested_end > start else source_duration_ms)
        if end <= start:
            continue
        director_duration = _number(item.get("directorDurationMs", item.get("director_duration_ms")), 0.0)
        normalized.append({
            "segmentId": str(item.get("segmentId", item.get("segment_id", ""))).strip(),
            "sourceTrimInMs": round(start),
            "sourceTrimOutMs": round(end),
            "directorDurationMs": round(director_duration) if director_duration > 0 else None,
            "index": index,
        })

    if not normalized:
        normalized = [{"segmentId": "", "sourceTrimInMs": 0, "sourceTrimOutMs": round(source_duration_ms), "directorDurationMs": None, "index": 0}]

    output_cursor = 0.0
    output_parts: list[dict[str, Any]] = []
    source_breaks: list[float] = []
    raw_source_breaks: list[float] = []
    timeline_parts: list[dict[str, float]] = []
    raw_cursor = 0.0
    output_timing_baked = False
    for index, part in enumerate(normalized):
        raw_duration = float(part["sourceTrimOutMs"] - part["sourceTrimInMs"])
        requested_director_duration = part.get("directorDurationMs")
        # Preserve live Director playback for non-VFI speed changes.  A 0.x
        # source is baked only when the existing slow-motion path requested it.
        # At normal 1x, bake the exact Director window so per-clip frame
        # rounding cannot accumulate down the timeline.
        bake_this_part = bool(timing_baked or (requested_director_duration and abs(speed - 1.0) < 1e-9))
        if requested_director_duration and bake_this_part:
            output_duration = float(requested_director_duration)
        elif timing_baked:
            output_duration = raw_duration / speed
        else:
            output_duration = raw_duration
        source_timeline_start = raw_cursor
        source_timeline_end = raw_cursor + raw_duration
        output_parts.append({
            "segmentId": part["segmentId"],
            "sourceTrimInMs": part["sourceTrimInMs"],
            "sourceTrimOutMs": part["sourceTrimOutMs"],
            "outputTrimInMs": round(output_cursor),
            "outputTrimOutMs": round(output_cursor + output_duration),
            "timingBaked": bake_this_part,
        })
        timeline_parts.append({
            "sourceStartMs": source_timeline_start,
            "sourceEndMs": source_timeline_end,
            "outputStartMs": output_cursor,
            "outputEndMs": output_cursor + output_duration,
            "timingBaked": float(bake_this_part),
        })
        raw_cursor += raw_duration
        if index < len(normalized) - 1:
            raw_source_breaks.append(raw_cursor)
            source_breaks.append(output_cursor + output_duration if bake_this_part else raw_cursor)
        output_cursor += output_duration
        output_timing_baked = output_timing_baked or bake_this_part

    is_full_source = len(normalized) == 1 and normalized[0]["sourceTrimInMs"] == 0 and abs(normalized[0]["sourceTrimOutMs"] - source_duration_ms) < 1
    if is_full_source and not output_timing_baked:
        return PreparedVideo(source, output_parts, source_breaks, raw_source_breaks, timeline_parts, False)

    filters: list[str] = []
    for index, part in enumerate(normalized):
        start = f"{part['sourceTrimInMs'] / 1000:.3f}"
        end = f"{part['sourceTrimOutMs'] / 1000:.3f}"
        raw_duration = max(0.001, float(part["sourceTrimOutMs"] - part["sourceTrimInMs"]) / 1000)
        target_duration = float(timeline_parts[index]["outputEndMs"] - timeline_parts[index]["outputStartMs"]) / 1000
        bake_this_part = bool(timeline_parts[index]["timingBaked"])
        video_filter = f"[0:v:0]trim=start={start}:end={end},setpts=PTS-STARTPTS"
        if bake_this_part:
            ratio = target_duration / raw_duration
            # tpad guarantees one complete frame covering the exact requested
            # Director endpoint even when the source decoder rounds down.
            video_filter += f",setpts=PTS*{ratio:.12f},tpad=stop_mode=clone:stop_duration=0.250,trim=duration={target_duration:.6f},setpts=PTS-STARTPTS"
        filters.append(f"{video_filter}[v{index}]")
        if has_audio:
            audio_filter = f"[0:a:0]atrim=start={start}:end={end},asetpts=PTS-STARTPTS"
            if bake_this_part:
                audio_filter += f",{_atempo_chain(raw_duration / target_duration)},apad=pad_dur=0.250,atrim=duration={target_duration:.6f},asetpts=PTS-STARTPTS"
            filters.append(f"{audio_filter}[a{index}]")
    if has_audio:
        inputs = "".join(f"[v{index}][a{index}]" for index in range(len(normalized)))
        filters.append(f"{inputs}concat=n={len(normalized)}:v=1:a=1[vout][aout]")
    else:
        inputs = "".join(f"[v{index}]" for index in range(len(normalized)))
        filters.append(f"{inputs}concat=n={len(normalized)}:v=1:a=0[vout]")
    args = ["ffmpeg", "-v", "error", "-i", str(source), "-filter_complex", ";".join(filters), "-map", "[vout]"]
    if has_audio:
        args += ["-map", "[aout]", "-c:a", "pcm_s16le"]
    args += ["-c:v", "ffv1", "-level", "3", "-y", str(destination)]
    subprocess.run(args, check=True, timeout=1200)
    return PreparedVideo(destination, output_parts, source_breaks, raw_source_breaks, timeline_parts, output_timing_baked)


def retime_video_to_director_parts(
    source: Path,
    destination: Path,
    prepared: PreparedVideo,
    *,
    input_timing_scale: float = 1.0,
) -> Path:
    """Bake a FlashVSR/VFI result back onto the requested Director windows.

    FlashVSR writes a fresh CFR file and therefore loses the selected source's
    temporary timestamps. This restores those timestamps without touching the
    already-upscaled pixels beyond the normal final encoder pass.
    """
    if not prepared.timing_baked:
        return source
    filters: list[str] = []
    scale = max(0.001, float(input_timing_scale))
    for index, part in enumerate(prepared.timeline_parts):
        source_start = float(part["sourceStartMs"]) / 1000 * scale
        source_end = float(part["sourceEndMs"]) / 1000 * scale
        target_duration = max(0.001, float(part["outputEndMs"] - part["outputStartMs"]) / 1000)
        raw_duration = max(0.001, source_end - source_start)
        ratio = target_duration / raw_duration
        filters.append(
            f"[0:v:0]trim=start={source_start:.6f}:end={source_end:.6f},setpts=PTS-STARTPTS,"
            f"setpts=PTS*{ratio:.12f},tpad=stop_mode=clone:stop_duration=0.250,"
            f"trim=duration={target_duration:.6f},setpts=PTS-STARTPTS[v{index}]"
        )
    inputs = "".join(f"[v{index}]" for index in range(len(prepared.timeline_parts)))
    filters.append(f"{inputs}concat=n={len(prepared.timeline_parts)}:v=1:a=0[vout]")
    subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(source), "-filter_complex", ";".join(filters),
        "-map", "[vout]", "-an", "-c:v", "ffv1", "-level", "3", "-y", str(destination)
    ], check=True, timeout=1200)
    return destination

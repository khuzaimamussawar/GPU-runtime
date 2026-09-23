from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PreparedVideo:
    path: Path
    parts: list[dict[str, Any]]
    source_breaks_ms: list[float]


def _number(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
        return parsed if parsed >= 0 else fallback
    except (TypeError, ValueError):
        return fallback


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
        normalized.append({
            "segmentId": str(item.get("segmentId", item.get("segment_id", ""))).strip(),
            "sourceTrimInMs": round(start),
            "sourceTrimOutMs": round(end),
            "index": index,
        })

    if not normalized:
        normalized = [{"segmentId": "", "sourceTrimInMs": 0, "sourceTrimOutMs": round(source_duration_ms), "index": 0}]

    output_cursor = 0.0
    divisor = speed if timing_baked else 1.0
    output_parts: list[dict[str, Any]] = []
    source_breaks: list[float] = []
    raw_cursor = 0.0
    for index, part in enumerate(normalized):
        raw_duration = float(part["sourceTrimOutMs"] - part["sourceTrimInMs"])
        output_duration = raw_duration / divisor
        output_parts.append({
            "segmentId": part["segmentId"],
            "sourceTrimInMs": part["sourceTrimInMs"],
            "sourceTrimOutMs": part["sourceTrimOutMs"],
            "outputTrimInMs": round(output_cursor),
            "outputTrimOutMs": round(output_cursor + output_duration),
            "timingBaked": bool(timing_baked),
        })
        raw_cursor += raw_duration
        if index < len(normalized) - 1:
            source_breaks.append(raw_cursor)
        output_cursor += output_duration

    is_full_source = len(normalized) == 1 and normalized[0]["sourceTrimInMs"] == 0 and abs(normalized[0]["sourceTrimOutMs"] - source_duration_ms) < 1
    if is_full_source:
        return PreparedVideo(source, output_parts, source_breaks)

    filters: list[str] = []
    for index, part in enumerate(normalized):
        start = f"{part['sourceTrimInMs'] / 1000:.3f}"
        end = f"{part['sourceTrimOutMs'] / 1000:.3f}"
        filters.append(f"[0:v:0]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{index}]")
        if has_audio:
            filters.append(f"[0:a:0]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{index}]")
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
    return PreparedVideo(destination, output_parts, source_breaks)

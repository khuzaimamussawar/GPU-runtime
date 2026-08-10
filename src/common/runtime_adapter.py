from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import src.common.h3_runtime as runtime
from src.common.job_contract import H3Job


# Keep the large, stable runtime module untouched while fixing production-specific
# workflow behavior here. The four provider handlers import run_h3_job from this
# adapter, so both RunPod and Novita execute the same H3 contract.
_ORIGINAL_PREPARE_WORKFLOW = runtime.prepare_workflow


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


def _round_h3_frames(job: H3Job) -> int:
    """Round up to MiniMax H3's native 17*k+5 frame lattice."""
    requested = max(1, round(job.duration_seconds * job.fps))
    if requested <= 5:
        return 5
    if (requested - 5) % 17 == 0:
        return requested
    k = max(0, math.ceil((requested - 5) / 17))
    return 17 * k + 5


def _prepare_workflow(
    workflow: dict[str, Any],
    manifest: dict[str, Any],
    job: H3Job,
) -> dict[str, Any]:
    prepared = _ORIGINAL_PREPARE_WORKFLOW(workflow, manifest, job)
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

        if last_frame:
            runtime.set_path(
                workflow,
                required["lastFrameImage"],
                runtime.materialize_image(last_frame, media_dir, job, "match"),
            )
        else:
            _remove_path(workflow, optional.get("lastFrameBinding"))
        return

    if job.task_family != "h3_ref2va":
        raise runtime.H3RuntimeError(f"Unsupported taskFamily: {job.task_family}")

    refs = _as_list(inputs.get("referenceImages") or inputs.get("references"))
    if not refs and first_frame:
        refs = [first_frame]

    video_refs = _as_list(inputs.get("referenceVideos") or inputs.get("referenceVideo") or inputs.get("video"))
    audio_refs = _as_list(inputs.get("referenceAudios") or inputs.get("referenceAudio") or inputs.get("audio"))

    if not refs and not video_refs and not audio_refs:
        raise runtime.H3RuntimeError(
            "h3_ref2va requires at least one image, video, or audio reference"
        )

    reference_fit = runtime.normalize_reference_fit(job.settings)

    # The baked master intentionally contains placeholder loaders so the graph is
    # easy to edit in ComfyUI. Remove every unused binding before submission so
    # example.png/test media can never leak into a production generation.
    for index in range(9):
        loader_path = optional.get(f"referenceImage{index}")
        binding_path = optional.get(f"referenceImageBinding{index}")
        if index < len(refs) and loader_path:
            runtime.set_path(
                workflow,
                loader_path,
                runtime.materialize_image(
                    runtime.media_ref(refs[index]), media_dir, job, reference_fit
                ),
            )
        else:
            _remove_path(workflow, binding_path)

    for index in range(3):
        loader_path = optional.get(f"referenceVideo{index}")
        video_binding = optional.get(f"referenceVideoBinding{index}")
        audio_binding = optional.get(f"referenceVideoAudioBinding{index}")
        if index < len(video_refs) and loader_path:
            runtime.set_path(
                workflow,
                loader_path,
                runtime.download_media(runtime.media_ref(video_refs[index]), media_dir),
            )
        else:
            _remove_path(workflow, video_binding)
            _remove_path(workflow, audio_binding)

    for index in range(3):
        loader_path = optional.get(f"referenceAudio{index}")
        binding_path = optional.get(f"referenceAudioBinding{index}")
        if index < len(audio_refs) and loader_path:
            runtime.set_path(
                workflow,
                loader_path,
                runtime.download_media(runtime.media_ref(audio_refs[index]), media_dir),
            )
        else:
            _remove_path(workflow, binding_path)


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
    if not outputs:
        return False
    return all(bool(item.get("uploaded")) for item in outputs.values() if isinstance(item, dict))


def run_h3_job(
    payload: dict[str, Any], expected_task_family: str, runtime_name: str
) -> dict[str, Any]:
    # Install deterministic process-wide patches once. They are identical for all
    # jobs in a worker, so concurrent imports do not change behavior per request.
    runtime.normalize_frame_count = _round_h3_frames
    runtime.materialize_inputs = _materialize_inputs
    runtime.prepare_workflow = _prepare_workflow

    normalized = runtime.unwrap_payload(payload)
    job_id = str(normalized.get("jobId") or normalized.get("job_id") or normalized.get("id") or "unknown")
    result: dict[str, Any] | None = None
    try:
        result = runtime.run_h3_job(payload, expected_task_family, runtime_name)
        return result
    finally:
        # Always remove downloaded inputs. Encoded/source outputs are removed only
        # after R2 upload succeeded, so a misconfigured storage endpoint still
        # leaves local files available for debugging instead of deleting results.
        _cleanup_job_files(job_id, remove_outputs=_all_outputs_uploaded(result))

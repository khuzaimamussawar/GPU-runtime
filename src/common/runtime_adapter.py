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

# The shared SageAttention 2.2.0 artifact is intentionally compiled only for the
# production Ada + Blackwell architectures. Do not let a UI toggle force Sage on
# for an unsupported worker (for example an unknown Hopper-backed MIG slice).
_SAGE_COMPILED_CAPABILITIES = {(8, 9), (12, 0)}


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


def _prepare_workflow(
    workflow: dict[str, Any],
    manifest: dict[str, Any],
    job: H3Job,
) -> dict[str, Any]:
    prepared = _ORIGINAL_PREPARE_WORKFLOW(workflow, manifest, job)
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
    audio_refs = _as_list(inputs.get("referenceAudios") or inputs.get("referenceAudio") or inputs.get("audio"))

    if len(refs) > 9:
        raise runtime.H3RuntimeError("h3_ref2va supports at most 9 image references per job")
    if len(video_refs) > 3:
        raise runtime.H3RuntimeError("h3_ref2va supports at most 3 reference videos per job")
    if len(audio_refs) > 3:
        raise runtime.H3RuntimeError("h3_ref2va supports at most 3 standalone audio references per job")
    if not refs and not video_refs and not audio_refs:
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
            runtime.download_media(runtime.media_ref(ref), media_dir),
        )

    for index, ref in enumerate(audio_refs):
        loader_path = optional.get(f"referenceAudio{index}")
        if not loader_path:
            raise runtime.H3RuntimeError(f"Missing workflow path for reference audio {index}")
        runtime.set_path(
            workflow,
            loader_path,
            runtime.download_media(runtime.media_ref(ref), media_dir),
        )

    _prune_ref2va_placeholders(
        workflow,
        optional,
        image_count=len(refs),
        video_count=len(video_refs),
        audio_count=len(audio_refs),
    )


def _upload_outputs(paths: dict[str, Path], job: H3Job) -> dict[str, Any]:
    """Upload the two final files using the SceneBuilder plan's canonical keys."""
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
    payload: dict[str, Any], expected_task_family: str, runtime_name: str
) -> dict[str, Any]:
    # Install deterministic process-wide patches once. They are identical for all
    # jobs in a worker, so concurrent imports do not change behavior per request.
    runtime.normalize_frame_count = _round_h3_frames
    runtime.materialize_inputs = _materialize_inputs
    runtime.prepare_workflow = _prepare_workflow
    runtime.upload_outputs = _upload_outputs

    normalized = runtime.unwrap_payload(payload)
    normalized.setdefault("taskFamily", expected_task_family)
    job_id = str(normalized.get("jobId") or normalized.get("job_id") or normalized.get("id") or "unknown")
    result: dict[str, Any] | None = None
    try:
        result = runtime.run_h3_job(payload, expected_task_family, runtime_name)
        # The base runtime reports raw requested frames; return the actual H3 frame
        # count used by the patched workflow so D1/debug metadata stays truthful.
        try:
            result["frames"] = _round_h3_frames(runtime.normalize_job(normalized))
        except Exception:
            pass
        return result
    finally:
        # Always remove downloaded inputs. Encoded/source outputs are removed only
        # after R2 upload succeeded, so a misconfigured storage endpoint still
        # leaves local files available for debugging instead of deleting results.
        _cleanup_job_files(job_id, remove_outputs=_all_outputs_uploaded(result))

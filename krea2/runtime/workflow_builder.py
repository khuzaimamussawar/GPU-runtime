from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable


class WorkflowBuildError(ValueError):
    pass


VAE_ALIASES = {
    "qwen_image": "qwen_image_vae.safetensors",
    "qwen": "qwen_image_vae.safetensors",
    "wan_2_1": "wan_2.1_vae.safetensors",
    "wan": "wan_2.1_vae.safetensors",
}

APPROVED_RENDER_SIZES = {
    (1280, 720),
    (720, 1280),
    (2048, 1152),
    (1152, 2048),
}
ALLOWED_SAMPLERS = {"euler"}
ALLOWED_SCHEDULERS = {"simple"}
MAX_SAFE_SEED = 9_007_199_254_740_991
MIN_CFG = 1.0
MAX_CFG = 1.5


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_parent(obj: dict[str, Any], dotted: str) -> tuple[Any, str | int]:
    parts = dotted.split(".")
    if not parts:
        raise WorkflowBuildError(f"invalid empty workflow path: {dotted!r}")

    current: Any = obj
    for part in parts[:-1]:
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError) as exc:
                raise WorkflowBuildError(f"workflow path does not resolve: {dotted}") from exc
        elif isinstance(current, dict):
            if part not in current:
                raise WorkflowBuildError(f"workflow path does not resolve: {dotted}")
            current = current[part]
        else:
            raise WorkflowBuildError(f"workflow path does not resolve: {dotted}")

    final: str | int
    if isinstance(current, list):
        try:
            final = int(parts[-1])
        except ValueError as exc:
            raise WorkflowBuildError(f"workflow path does not resolve: {dotted}") from exc
    else:
        final = parts[-1]
    return current, final


def _set_path(obj: dict[str, Any], dotted: str, value: Any) -> None:
    parent, key = _resolve_parent(obj, dotted)
    if isinstance(parent, list):
        if not isinstance(key, int) or key < 0 or key >= len(parent):
            raise WorkflowBuildError(f"workflow path does not resolve: {dotted}")
        parent[key] = value
    elif isinstance(parent, dict):
        if key not in parent:
            raise WorkflowBuildError(f"workflow path does not resolve: {dotted}")
        parent[key] = value
    else:
        raise WorkflowBuildError(f"workflow path does not resolve: {dotted}")


def _required_path(manifest: dict[str, Any], key: str) -> str:
    path = (manifest.get("requiredPaths") or {}).get(key)
    if not path:
        raise WorkflowBuildError(f"manifest missing required path {key!r}")
    return str(path)


def _optional_path(manifest: dict[str, Any], key: str) -> str | None:
    path = (manifest.get("optionalPaths") or {}).get(key)
    return str(path) if path else None


def _as_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    raise WorkflowBuildError(f"{field} must be boolean")


def _as_int(value: Any, *, field: str, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool):
        raise WorkflowBuildError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise WorkflowBuildError(f"{field} must be an integer") from exc
    if minimum is not None and parsed < minimum:
        raise WorkflowBuildError(f"{field} must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise WorkflowBuildError(f"{field} must be <= {maximum}")
    return parsed


def _as_float(value: Any, *, field: str, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool):
        raise WorkflowBuildError(f"{field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise WorkflowBuildError(f"{field} must be numeric") from exc
    if minimum is not None and parsed < minimum:
        raise WorkflowBuildError(f"{field} must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise WorkflowBuildError(f"{field} must be <= {maximum}")
    return parsed


def _clean_string(value: Any, *, field: str, allow_empty: bool = False) -> str:
    text = str(value or "").strip()
    if not text and not allow_empty:
        raise WorkflowBuildError(f"{field} must not be empty")
    return text


def _patch_common(workflow: dict[str, Any], manifest: dict[str, Any], settings: dict[str, Any]) -> None:
    if "prompt" in settings:
        _set_path(workflow, _required_path(manifest, "prompt"), str(settings.get("prompt") or ""))

    # Krea prompt enhancement is intentionally disabled. SceneBuilder shapes the
    # effective prompt before dispatch, and the pod must never expand it again.
    prompt_enhance_path = (manifest.get("requiredPaths") or {}).get("promptEnhance")
    if prompt_enhance_path:
        _set_path(workflow, str(prompt_enhance_path), False)

    width = settings.get("width")
    height = settings.get("height")
    if (width is None) != (height is None):
        raise WorkflowBuildError("width and height must be supplied together")
    if width is not None and height is not None:
        width = _as_int(width, field="width")
        height = _as_int(height, field="height")
        if (width, height) not in APPROVED_RENDER_SIZES:
            allowed = ", ".join(f"{w}x{h}" for w, h in sorted(APPROVED_RENDER_SIZES))
            raise WorkflowBuildError(f"unsupported Krea render size {width}x{height}; allowed: {allowed}")
        for key in ("width", "widthLatent", "widthSampling"):
            path = (manifest.get("requiredPaths") or {}).get(key)
            if path:
                _set_path(workflow, str(path), width)
        for key in ("height", "heightLatent", "heightSampling"):
            path = (manifest.get("requiredPaths") or {}).get(key)
            if path:
                _set_path(workflow, str(path), height)

    if "vae" in settings:
        raw_vae = _clean_string(settings.get("vae"), field="vae")
        vae_file = VAE_ALIASES.get(raw_vae, raw_vae)
        allowed = set((manifest.get("modelFiles") or {}).get("vaeOptions", {}).values())
        if allowed and vae_file not in allowed:
            raise WorkflowBuildError(f"unsupported VAE: {raw_vae}")
        _set_path(workflow, _required_path(manifest, "vaeName"), vae_file)

    if "seed" in settings:
        _set_path(
            workflow,
            _required_path(manifest, "seed"),
            _as_int(settings.get("seed"), field="seed", minimum=0, maximum=MAX_SAFE_SEED),
        )
    if "steps" in settings:
        _set_path(
            workflow,
            _required_path(manifest, "steps"),
            _as_int(settings.get("steps"), field="steps", minimum=4, maximum=12),
        )
    if "cfg" in settings:
        _set_path(
            workflow,
            _required_path(manifest, "cfg"),
            _as_float(settings.get("cfg"), field="cfg", minimum=MIN_CFG, maximum=MAX_CFG),
        )
    if "sampler" in settings:
        sampler = _clean_string(settings.get("sampler"), field="sampler")
        if sampler not in ALLOWED_SAMPLERS:
            raise WorkflowBuildError(f"unsupported sampler: {sampler}")
        _set_path(workflow, _required_path(manifest, "sampler"), sampler)
    if "scheduler" in settings:
        scheduler = _clean_string(settings.get("scheduler"), field="scheduler")
        if scheduler not in ALLOWED_SCHEDULERS:
            raise WorkflowBuildError(f"unsupported scheduler: {scheduler}")
        _set_path(workflow, _required_path(manifest, "scheduler"), scheduler)
    if "denoise" in settings:
        _set_path(
            workflow,
            _required_path(manifest, "denoise"),
            _as_float(settings.get("denoise"), field="denoise", minimum=0.0, maximum=1.0),
        )

    output_prefix = settings.get("outputPrefix")
    output_path = _optional_path(manifest, "outputPrefix")
    if output_prefix is not None and output_path:
        _set_path(workflow, output_path, _clean_string(output_prefix, field="outputPrefix"))


def _normalize_loras(user_loras: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in user_loras or []:
        item = dict(raw)
        file_name = _clean_string(item.get("fileName") or item.get("file_name"), field="LoRA fileName")
        strength = _as_float(item.get("strength"), field=f"LoRA strength ({file_name})")
        min_strength = item.get("minStrength", item.get("min_strength"))
        max_strength = item.get("maxStrength", item.get("max_strength"))
        if min_strength is None or max_strength is None:
            raise WorkflowBuildError(f"LoRA catalog range is required for {file_name}")
        min_strength = _as_float(min_strength, field=f"LoRA minStrength ({file_name})")
        max_strength = _as_float(max_strength, field=f"LoRA maxStrength ({file_name})")
        if min_strength > max_strength:
            raise WorkflowBuildError(f"LoRA minimum exceeds maximum for {file_name}")
        if strength < min_strength or strength > max_strength:
            raise WorkflowBuildError(
                f"INVALID_LORA_STRENGTH: {strength} outside [{min_strength}, {max_strength}] for {file_name}"
            )
        normalized.append(
            {
                "loraId": str(item.get("loraId") or item.get("lora_id") or ""),
                "fileName": file_name,
                "strength": strength,
            }
        )
    return normalized


def _patch_user_loras(
    workflow: dict[str, Any],
    manifest: dict[str, Any],
    user_loras: Iterable[dict[str, Any]] | None,
) -> None:
    config = ((manifest.get("runtimeGraphPatching") or {}).get("userLoras") or {})
    max_count = int(config.get("maxCount", 0))
    loras = _normalize_loras(user_loras)
    if len(loras) > max_count:
        raise WorkflowBuildError(f"this workflow supports at most {max_count} user LoRAs")

    if max_count == 0 and loras:
        raise WorkflowBuildError("user LoRAs are not enabled for this workflow")
    if max_count == 0:
        return

    base_node = str(config.get("baseModelNode") or "")
    consumer_node = str(config.get("consumerNode") or "")
    consumer_input = str(config.get("consumerInput") or "model")
    loader_class = str(config.get("loaderClass") or "LoraLoaderModelOnly")
    name_input = str(config.get("nameInput") or "lora_name")
    strength_input = str(config.get("strengthInput") or "strength_model")
    if not base_node or base_node not in workflow or not consumer_node or consumer_node not in workflow:
        raise WorkflowBuildError("LoRA graph patch manifest is invalid")

    for index in range(1, max_count + 1):
        workflow.pop(f"sb_lora_{index:02d}", None)
    workflow[consumer_node].setdefault("inputs", {})[consumer_input] = [base_node, 0]

    previous_node = base_node
    for index, lora in enumerate(loras, start=1):
        node_id = f"sb_lora_{index:02d}"
        workflow[node_id] = {
            "inputs": {
                name_input: lora["fileName"],
                strength_input: lora["strength"],
                "model": [previous_node, 0],
            },
            "class_type": loader_class,
            "_meta": {
                "title": f"SceneBuilder User LoRA {index}",
                "lora_id": lora["loraId"],
            },
        }
        previous_node = node_id

    workflow[consumer_node]["inputs"][consumer_input] = [previous_node, 0]


def _patch_style_references(
    workflow: dict[str, Any],
    manifest: dict[str, Any],
    reference_images: Iterable[str] | None,
) -> None:
    config = ((manifest.get("runtimeGraphPatching") or {}).get("styleReferences") or {})
    min_count = int(config.get("minCount", 0))
    max_count = int(config.get("maxCount", 0))
    refs = [_clean_string(value, field="style reference image") for value in (reference_images or [])]

    if len(refs) < min_count or len(refs) > max_count:
        raise WorkflowBuildError(
            f"style-reference mode requires {min_count}-{max_count} reference images"
        )

    encoder_node = str(config.get("encoderNode") or "")
    first_loader = str(config.get("firstLoaderNode") or "")
    loader_class = str(config.get("loaderClass") or "LoadImage")
    input_prefix = str(config.get("inputPrefix") or "image")
    if not encoder_node or encoder_node not in workflow or not first_loader or first_loader not in workflow:
        raise WorkflowBuildError("style-reference graph patch manifest is invalid")

    encoder_inputs = workflow[encoder_node].setdefault("inputs", {})
    for index in range(2, max_count + 1):
        workflow.pop(f"sb_ref_{index:02d}", None)
    for index in range(1, max_count + 1):
        encoder_inputs.pop(f"{input_prefix}{index}", None)

    workflow[first_loader].setdefault("inputs", {})["image"] = refs[0]
    encoder_inputs[f"{input_prefix}1"] = [first_loader, 0]

    for index, filename in enumerate(refs[1:], start=2):
        node_id = f"sb_ref_{index:02d}"
        workflow[node_id] = {
            "inputs": {"image": filename},
            "class_type": loader_class,
            "_meta": {"title": f"SceneBuilder Style Reference {index}"},
        }
        encoder_inputs[f"{input_prefix}{index}"] = [node_id, 0]


def _patch_negative_prompt(
    workflow: dict[str, Any],
    manifest: dict[str, Any],
    settings: dict[str, Any],
) -> None:
    config = ((manifest.get("runtimeGraphPatching") or {}).get("negativePrompt") or {})
    if not config:
        return

    consumer_node = str(config.get("consumerNode") or "")
    consumer_input = str(config.get("consumerInput") or "negative")
    zero_node = str(config.get("zeroNode") or "")
    clip_node = str(config.get("clipNode") or "")
    if not consumer_node or consumer_node not in workflow or not zero_node or zero_node not in workflow:
        raise WorkflowBuildError("negative-prompt graph patch manifest is invalid")
    if not clip_node or clip_node not in workflow:
        raise WorkflowBuildError("negative-prompt clip node is invalid")

    workflow.pop("sb_negative", None)
    workflow[consumer_node].setdefault("inputs", {})[consumer_input] = [zero_node, 0]

    negative = str(settings.get("negativePrompt", settings.get("negative_prompt", "")) or "").strip()
    cfg = _as_float(settings.get("cfg", 1.0), field="cfg", minimum=MIN_CFG, maximum=MAX_CFG)
    if not negative or cfg <= 1.0:
        return

    workflow["sb_negative"] = {
        "inputs": {
            "text": negative,
            "clip": [clip_node, 0],
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "SceneBuilder Guided Negative Prompt"},
    }
    workflow[consumer_node]["inputs"][consumer_input] = ["sb_negative", 0]


def prepare_krea2_workflow(
    workflow: dict[str, Any],
    manifest: dict[str, Any],
    *,
    settings: dict[str, Any] | None = None,
    user_loras: Iterable[dict[str, Any]] | None = None,
    reference_images: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return a validated, patched Comfy API workflow for one Krea 2 image job."""

    prepared = copy.deepcopy(workflow)
    settings = dict(settings or {})
    mode = str(manifest.get("mode") or "").strip()
    if mode not in {"lora", "reference_images"}:
        raise WorkflowBuildError(f"unsupported Krea2 workflow mode: {mode or '<missing>'}")

    _patch_common(prepared, manifest, settings)

    if mode == "lora":
        if list(reference_images or []):
            raise WorkflowBuildError("style references require the reference_images workflow")
        _patch_user_loras(prepared, manifest, user_loras)
    else:
        _patch_user_loras(prepared, manifest, user_loras)
        _patch_style_references(prepared, manifest, reference_images)

    _patch_negative_prompt(prepared, manifest, settings)
    return prepared


def load_and_prepare(
    workflow_path: str | Path,
    manifest_path: str | Path,
    *,
    settings: dict[str, Any] | None = None,
    user_loras: Iterable[dict[str, Any]] | None = None,
    reference_images: Iterable[str] | None = None,
) -> dict[str, Any]:
    workflow = _load_json(Path(workflow_path))
    manifest = _load_json(Path(manifest_path))
    return prepare_krea2_workflow(
        workflow,
        manifest,
        settings=settings,
        user_loras=user_loras,
        reference_images=reference_images,
    )

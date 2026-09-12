from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable

from krea2.runtime.workflow_builder import load_and_prepare
from src.image_pod.media import (
    cleanup_job_inputs,
    finalize_image_outputs,
    materialize_user_loras,
    stage_style_references,
)


ProgressCallback = Callable[[str, int | None], None]


def _raw_lora_items(values: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [dict(value) for value in (values or [])]


def _materialize_user_loras_with_logging(
    job_id: str,
    values: Iterable[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    raw = _raw_lora_items(values)
    print(f"[Krea2 LoRA] job={job_id} received count={len(raw)}", flush=True)
    for index, item in enumerate(raw, start=1):
        lora_id = str(item.get("loraId") or item.get("lora_id") or "").strip()
        object_key = str(item.get("objectKey") or item.get("r2ObjectKey") or "").strip()
        sha256 = str(item.get("sha256") or "").strip().lower()
        strength = item.get("strength")
        expected_size = item.get("fileSizeBytes", item.get("file_size_bytes"))
        print(
            f"[Krea2 LoRA] job={job_id} materialize index={index} id={lora_id} "
            f"strength={strength} objectKey={object_key} bytes={expected_size} sha256={sha256}",
            flush=True,
        )

    try:
        resolved = materialize_user_loras(raw)
    except Exception as exc:
        print(
            f"[Krea2 LoRA] job={job_id} materialization FAILED count={len(raw)} "
            f"error={type(exc).__name__}: {exc}",
            flush=True,
        )
        raise

    for index, item in enumerate(resolved, start=1):
        print(
            f"[Krea2 LoRA] job={job_id} materialized index={index} "
            f"id={item.get('loraId')} file={item.get('fileName')} strength={item.get('strength')}",
            flush=True,
        )
    return resolved


def _verify_lora_graph(
    job_id: str,
    workflow: dict[str, Any],
    manifest: dict[str, Any],
    user_loras: Iterable[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    loras = _raw_lora_items(user_loras)
    if not loras:
        return []

    config = ((manifest.get("runtimeGraphPatching") or {}).get("userLoras") or {})
    base_node = str(config.get("baseModelNode") or "")
    consumer_node = str(config.get("consumerNode") or "")
    consumer_input = str(config.get("consumerInput") or "model")
    loader_class = str(config.get("loaderClass") or "LoraLoaderModelOnly")
    name_input = str(config.get("nameInput") or "lora_name")
    strength_input = str(config.get("strengthInput") or "strength_model")
    if not base_node or not consumer_node:
        raise RuntimeError("Krea2 LoRA graph audit manifest is missing base/consumer nodes")

    loaded: list[dict[str, Any]] = []
    previous_node = base_node
    for index, expected in enumerate(loras, start=1):
        node_id = f"sb_lora_{index:02d}"
        node = workflow.get(node_id)
        if not isinstance(node, dict):
            raise RuntimeError(f"Krea2 LoRA graph missing expected node {node_id}")
        if str(node.get("class_type") or "") != loader_class:
            raise RuntimeError(
                f"Krea2 LoRA graph node {node_id} class mismatch: {node.get('class_type')} != {loader_class}"
            )

        inputs = node.get("inputs") or {}
        expected_file = str(expected.get("fileName") or "")
        actual_file = str(inputs.get(name_input) or "")
        if actual_file != expected_file:
            raise RuntimeError(
                f"Krea2 LoRA graph node {node_id} file mismatch: {actual_file} != {expected_file}"
            )

        try:
            expected_strength = float(expected.get("strength"))
            actual_strength = float(inputs.get(strength_input))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Krea2 LoRA graph node {node_id} has invalid strength") from exc
        if actual_strength != expected_strength:
            raise RuntimeError(
                f"Krea2 LoRA graph node {node_id} strength mismatch: {actual_strength} != {expected_strength}"
            )

        expected_source = [previous_node, 0]
        if inputs.get("model") != expected_source:
            raise RuntimeError(
                f"Krea2 LoRA graph node {node_id} model source mismatch: "
                f"{inputs.get('model')} != {expected_source}"
            )

        entry = {
            "index": index,
            "nodeId": node_id,
            "loraId": str(expected.get("loraId") or ""),
            "fileName": actual_file,
            "strength": actual_strength,
            "modelSource": previous_node,
        }
        loaded.append(entry)
        print(
            f"[Krea2 LoRA] job={job_id} graph index={index} node={node_id} "
            f"id={entry['loraId']} file={actual_file} strength={actual_strength} "
            f"modelSource={previous_node}",
            flush=True,
        )
        previous_node = node_id

    consumer = workflow.get(consumer_node)
    consumer_inputs = consumer.get("inputs") if isinstance(consumer, dict) else None
    expected_final = [previous_node, 0]
    if not isinstance(consumer_inputs, dict) or consumer_inputs.get(consumer_input) != expected_final:
        actual = consumer_inputs.get(consumer_input) if isinstance(consumer_inputs, dict) else None
        raise RuntimeError(
            f"Krea2 LoRA graph consumer mismatch: {actual} != {expected_final}"
        )

    print(
        f"[Krea2 LoRA] job={job_id} graph verified count={len(loaded)} "
        f"consumer={consumer_node}.{consumer_input} finalNode={previous_node}",
        flush=True,
    )
    return loaded


class Krea2Adapter:
    task_family = "krea2_image"

    def __init__(self) -> None:
        self.root = Path(os.environ.get("SCENEBUILDER_IMAGE_ROOT", "/opt/scenebuilder-image"))
        self.model_root = Path(os.environ.get("SCENEBUILDER_KREA2_MODEL_ROOT", "/opt/scenebuilder-models/krea2"))
        self.comfy_root = Path(os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))
        self.comfy_host = os.environ.get("COMFY_HOST", "127.0.0.1")
        self.comfy_port = int(os.environ.get("COMFY_PORT", "8188"))
        self.comfy_url = f"http://{self.comfy_host}:{self.comfy_port}"
        self.extra_model_paths = Path(
            os.environ.get(
                "COMFY_EXTRA_MODEL_PATHS",
                str(self.root / "extra_model_paths.yaml"),
            )
        )
        self.startup_timeout = int(os.environ.get("COMFY_STARTUP_TIMEOUT_SECONDS", "240"))
        self.poll_interval = float(os.environ.get("COMFY_HISTORY_POLL_SECONDS", "1.0"))
        self._process: subprocess.Popen[Any] | None = None
        self._process_lock = threading.RLock()

    @property
    def workflows_root(self) -> Path:
        return self.root / "workflows"

    def _required_assets(self) -> list[Path]:
        return [
            self.model_root / "diffusion_models" / "krea2_turbo_int8_convrot.safetensors",
            self.model_root / "text_encoders" / "qwen3vl_4b_bf16.safetensors",
            self.model_root / "vae" / "qwen_image_vae.safetensors",
            self.model_root / "vae" / "wan_2.1_vae.safetensors",
            self.model_root / "loras" / "krea2_style_reference.safetensors",
            self.workflows_root / "krea2_turbo.json",
            self.workflows_root / "krea2_style_reference.json",
            self.workflows_root / "manifests" / "krea2_turbo.json",
            self.workflows_root / "manifests" / "krea2_style_reference.json",
            self.extra_model_paths,
        ]

    def readiness(self) -> dict[str, Any]:
        missing = [str(path) for path in self._required_assets() if not path.is_file() or path.stat().st_size <= 0]
        comfy_ready = self.is_ready()
        return {
            "ready": comfy_ready and not missing,
            "taskFamily": self.task_family,
            "comfyReady": comfy_ready,
            "missingAssets": missing,
            "modelRoot": str(self.model_root),
            "workflowsRoot": str(self.workflows_root),
        }

    def diagnostics(self) -> dict[str, Any]:
        details = self.readiness()
        details.update(
            {
                "comfyRoot": str(self.comfy_root),
                "comfyUrl": self.comfy_url,
                "extraModelPaths": str(self.extra_model_paths),
            }
        )
        return details

    def is_ready(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.comfy_url}/system_stats", timeout=2) as response:
                response.read(1024)
            return True
        except Exception:
            return False

    def ensure_ready(self) -> None:
        if self.is_ready():
            return

        with self._process_lock:
            if self.is_ready():
                return
            if self._process is not None and self._process.poll() is None:
                process = self._process
            else:
                cmd = [
                    "python3",
                    "main.py",
                    "--listen",
                    self.comfy_host,
                    "--port",
                    str(self.comfy_port),
                    "--disable-auto-launch",
                    "--extra-model-paths-config",
                    str(self.extra_model_paths),
                ]
                process = subprocess.Popen(cmd, cwd=self.comfy_root)
                self._process = process

        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.is_ready():
                return
            if process.poll() is not None:
                raise RuntimeError(f"ComfyUI exited during startup with code {process.returncode}")
            time.sleep(1.0)
        raise RuntimeError("ComfyUI did not become ready before startup timeout")

    def cancel(self) -> None:
        request = urllib.request.Request(
            f"{self.comfy_url}/interrupt",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                response.read(1024)
        except Exception:
            pass

    def run_job(
        self,
        payload: dict[str, Any],
        *,
        progress: ProgressCallback | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        self.ensure_ready()
        progress = progress or (lambda _phase, _percent=None: None)
        cancel_requested = cancel_requested or (lambda: False)

        job_id = str(payload.get("jobId") or payload.get("job_id") or "").strip()
        if not job_id:
            raise ValueError("jobId is required")
        project_id = str(payload.get("projectId") or payload.get("project_id") or "").strip() or None

        settings = self._normalize_settings(payload)
        style_mode = str(
            payload.get("styleMode")
            or settings.get("styleMode")
            or "lora"
        ).strip()
        if style_mode not in {"lora", "reference_images"}:
            raise ValueError("styleMode must be 'lora' or 'reference_images'")
        settings["styleMode"] = style_mode

        settings["outputPrefix"] = f"scenebuilder/{job_id}/image"
        r2_output_prefix = str(
            payload.get("outputPrefix")
            or payload.get("output_prefix")
            or settings.get("r2OutputPrefix")
            or ""
        ).strip("/") or None

        raw_user_loras = payload.get("userLoras") or payload.get("loras") or []
        staged_input_dir: Path | None = None
        if style_mode == "lora":
            workflow_path = self.workflows_root / "krea2_turbo.json"
            manifest_path = self.workflows_root / "manifests" / "krea2_turbo.json"
            progress("preparing_model", 3)
            user_loras = _materialize_user_loras_with_logging(job_id, raw_user_loras)
            reference_images: list[str] = []
        else:
            workflow_path = self.workflows_root / "krea2_style_reference.json"
            manifest_path = self.workflows_root / "manifests" / "krea2_style_reference.json"
            progress("preparing_model", 2)
            user_loras = _materialize_user_loras_with_logging(job_id, raw_user_loras)
            progress("encoding_prompt", 3)
            reference_images, staged_input_dir = stage_style_references(
                job_id,
                payload.get("referenceImages") or payload.get("styleReferences") or [],
            )

        try:
            progress("preparing_model", 5)
            workflow = load_and_prepare(
                workflow_path,
                manifest_path,
                settings=settings,
                user_loras=user_loras,
                reference_images=reference_images,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            loaded_loras = _verify_lora_graph(job_id, workflow, manifest, user_loras)

            if cancel_requested():
                raise RuntimeError("job cancelled before Comfy submission")

            progress("generating", 10)
            prompt_id = self._submit_prompt(workflow)
            print(
                f"[Krea2 LoRA] job={job_id} submitted promptId={prompt_id} count={len(loaded_loras)}",
                flush=True,
            )
            history = self._wait_for_history(prompt_id, progress=progress, cancel_requested=cancel_requested)
            comfy_outputs = self._collect_images(history)
            if not comfy_outputs:
                raise RuntimeError("ComfyUI completed without a SaveImage output")
            if loaded_loras:
                print(
                    f"[Krea2 LoRA] job={job_id} execution completed count={len(loaded_loras)} "
                    f"ids={','.join(item['loraId'] for item in loaded_loras)}",
                    flush=True,
                )

            progress("resizing", 92)
            finalized = finalize_image_outputs(
                job_id=job_id,
                project_id=project_id,
                outputs=comfy_outputs,
                settings=settings,
                output_prefix=r2_output_prefix,
            )
            progress("finalizing", 97)
            progress("completed", 100)
            return {
                "ok": True,
                "jobId": job_id,
                "projectId": project_id,
                "taskFamily": self.task_family,
                "styleMode": style_mode,
                "promptId": prompt_id,
                "image": finalized,
                "execution": {
                    "referenceCount": len(reference_images),
                    "userLoraCount": len(user_loras),
                    "loadedLoras": loaded_loras,
                    "vae": settings.get("vae", "qwen_image"),
                    "renderWidth": settings.get("width"),
                    "renderHeight": settings.get("height"),
                    "outputWidth": settings.get("outputWidth"),
                    "outputHeight": settings.get("outputHeight"),
                    "seedMode": settings.get("seedMode"),
                    "seed": settings.get("seed"),
                },
            }
        finally:
            cleanup_job_inputs(staged_input_dir)

    def _normalize_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = dict(payload.get("settings") or {})
        if "prompt" not in settings and "prompt" in payload:
            settings["prompt"] = payload.get("prompt")
        if "negativePrompt" not in settings:
            negative = payload.get("negativePrompt", payload.get("negative_prompt"))
            if negative is not None:
                settings["negativePrompt"] = negative
        if "width" not in settings:
            settings["width"] = settings.get("renderWidth", payload.get("renderWidth", payload.get("width")))
        if "height" not in settings:
            settings["height"] = settings.get("renderHeight", payload.get("renderHeight", payload.get("height")))
        if "outputWidth" not in settings and payload.get("outputWidth") is not None:
            settings["outputWidth"] = payload.get("outputWidth")
        if "outputHeight" not in settings and payload.get("outputHeight") is not None:
            settings["outputHeight"] = payload.get("outputHeight")
        if "seed" not in settings and "seed" in payload:
            settings["seed"] = payload.get("seed")

        raw_seed_mode = settings.get("seedMode", payload.get("seedMode"))
        if raw_seed_mode is not None:
            seed_mode = str(raw_seed_mode).strip().lower()
            if seed_mode not in {"random", "fixed"}:
                raise ValueError("seedMode must be 'random' or 'fixed'")
        else:
            seed_mode = None

        seed_value = settings.get("seed")
        has_explicit_seed = seed_value is not None and not (
            isinstance(seed_value, str) and not seed_value.strip()
        )
        if not has_explicit_seed:
            raise ValueError("seed is required; SceneBuilder must resolve random seed before pod submission")

        # SceneBuilder owns seed resolution. The pod must consume the exact durable
        # seed it receives and must never generate or replace one.
        settings["seedMode"] = seed_mode or "fixed"

        if settings.get("width") is None:
            settings.pop("width", None)
        if settings.get("height") is None:
            settings.pop("height", None)
        return settings

    def _submit_prompt(self, workflow: dict[str, Any]) -> str:
        body = json.dumps({"prompt": workflow}, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            f"{self.comfy_url}/prompt",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ComfyUI rejected workflow: HTTP {exc.code}: {detail}") from exc
        prompt_id = str(payload.get("prompt_id") or "").strip()
        if not prompt_id:
            raise RuntimeError(f"ComfyUI /prompt response missing prompt_id: {payload}")
        return prompt_id

    def _wait_for_history(
        self,
        prompt_id: str,
        *,
        progress: ProgressCallback,
        cancel_requested: Callable[[], bool],
    ) -> dict[str, Any]:
        deadline = time.time() + int(os.environ.get("KREA2_JOB_TIMEOUT_SECONDS", "900"))
        last_percent = 10
        while time.time() < deadline:
            if cancel_requested():
                self.cancel()
                raise RuntimeError("job cancelled")

            try:
                with urllib.request.urlopen(f"{self.comfy_url}/history/{prompt_id}", timeout=10) as response:
                    history = json.loads(response.read().decode("utf-8"))
            except Exception:
                history = {}

            entry = history.get(prompt_id)
            if isinstance(entry, dict):
                status = entry.get("status") or {}
                status_str = str(status.get("status_str") or "").lower()
                if status_str == "error":
                    raise RuntimeError(f"ComfyUI generation failed: {status}")
                if entry.get("outputs"):
                    return entry

            last_percent = min(90, last_percent + 1)
            progress("generating", last_percent)
            time.sleep(self.poll_interval)

        self.cancel()
        raise TimeoutError(f"Krea2 job timed out waiting for Comfy history: {prompt_id}")

    def _collect_images(self, history_entry: dict[str, Any]) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        outputs = history_entry.get("outputs") or {}
        for node_id, node_output in outputs.items():
            if not isinstance(node_output, dict):
                continue
            for image in node_output.get("images") or []:
                if not isinstance(image, dict):
                    continue
                collected.append(
                    {
                        "nodeId": str(node_id),
                        "filename": image.get("filename"),
                        "subfolder": image.get("subfolder") or "",
                        "type": image.get("type") or "output",
                    }
                )
        return collected

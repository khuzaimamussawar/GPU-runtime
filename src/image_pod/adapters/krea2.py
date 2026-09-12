from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from krea2.runtime.workflow_builder import WorkflowBuildError, load_and_prepare


ProgressCallback = Callable[[str, int | None], None]


class Krea2Adapter:
    task_family = "krea2_image"

    def __init__(self) -> None:
        self.root = Path(os.environ.get("SCENEBUILDER_IMAGE_ROOT", "/opt/scenebuilder-image"))
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

    def diagnostics(self) -> dict[str, Any]:
        return {
            "taskFamily": self.task_family,
            "comfyRoot": str(self.comfy_root),
            "comfyUrl": self.comfy_url,
            "extraModelPaths": str(self.extra_model_paths),
            "workflowsRoot": str(self.workflows_root),
            "ready": self.is_ready(),
        }

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

        style_mode = str(
            payload.get("styleMode")
            or (payload.get("settings") or {}).get("styleMode")
            or "lora"
        ).strip()
        if style_mode not in {"lora", "reference_images"}:
            raise ValueError("styleMode must be 'lora' or 'reference_images'")

        settings = dict(payload.get("settings") or {})
        settings.setdefault("outputPrefix", f"scenebuilder/{job_id}/image")

        if style_mode == "lora":
            workflow_path = self.workflows_root / "krea2_turbo.json"
            manifest_path = self.workflows_root / "manifests" / "krea2_turbo.json"
            user_loras = list(payload.get("userLoras") or payload.get("loras") or [])
            reference_images: list[str] = []
        else:
            workflow_path = self.workflows_root / "krea2_style_reference.json"
            manifest_path = self.workflows_root / "manifests" / "krea2_style_reference.json"
            user_loras = []
            reference_images = self._reference_filenames(payload.get("referenceImages") or payload.get("styleReferences") or [])

        progress("preparing_model", 5)
        workflow = load_and_prepare(
            workflow_path,
            manifest_path,
            settings=settings,
            user_loras=user_loras,
            reference_images=reference_images,
        )

        if cancel_requested():
            raise RuntimeError("job cancelled before Comfy submission")

        progress("generating", 10)
        prompt_id = self._submit_prompt(workflow)
        history = self._wait_for_history(prompt_id, progress=progress, cancel_requested=cancel_requested)
        progress("finalizing", 95)

        outputs = self._collect_images(history)
        if not outputs:
            raise RuntimeError("ComfyUI completed without a SaveImage output")

        progress("completed", 100)
        return {
            "ok": True,
            "jobId": job_id,
            "taskFamily": self.task_family,
            "styleMode": style_mode,
            "promptId": prompt_id,
            "outputs": outputs,
        }

    def _reference_filenames(self, values: Any) -> list[str]:
        if not isinstance(values, (list, tuple)):
            values = [values]
        filenames: list[str] = []
        for value in values:
            if isinstance(value, str):
                filename = value.strip()
            elif isinstance(value, dict):
                filename = str(
                    value.get("inputFilename")
                    or value.get("input_filename")
                    or value.get("filename")
                    or ""
                ).strip()
            else:
                filename = ""
            if not filename:
                raise ValueError(
                    "each style reference must already be staged in the Comfy input directory and provide inputFilename"
                )
            filenames.append(filename)
        return filenames

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

            # Without a websocket progress channel, expose bounded heartbeat progress.
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

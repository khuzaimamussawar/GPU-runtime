from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from .r2_store import download_file, upload_file

Progress = Callable[[str, float, dict[str, Any] | None], None]

ENGINE_MODELS = {"realesr-animevideov3", "realesr-general-x4v3", "rife-4.9"}
PROFILE_SHAPES: dict[str, tuple[int, int]] = {
    "854x480 landscape": (480, 854),
    "1056x594 landscape": (594, 1056),
    "1280x720 landscape": (720, 1280),
    "480x854 portrait": (854, 480),
    "594x1056 portrait": (1056, 594),
    "720x1280 portrait": (1280, 720),
    "1080 class": (1080, 1920),
    "1440 class": (1440, 2560),
    "2160 class": (2160, 3840),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def _run(args: list[str], timeout: int, debug: list[str]) -> str:
    debug.append("$ " + " ".join(args))
    proc = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    output = proc.stdout[-12000:]
    debug.extend(output.splitlines()[-80:])
    if proc.returncode:
        raise RuntimeError(f"TRT_BUILD_FAILED:command exited {proc.returncode}: {output[-1200:]}")
    return output


def _trtexec() -> str | None:
    return shutil.which("trtexec")


def _model(job: dict[str, Any]) -> str:
    raw = str(job.get("modelFamily") or job.get("model") or job.get("settings", {}).get("modelFamily") or "").strip()
    if raw.lower() in {"rife", "rife 4.9"}:
        raw = "rife-4.9"
    if raw not in ENGINE_MODELS:
        raise RuntimeError(f"TRT_BUILD_FAILED:unsupported engine model {raw}")
    return raw


def _onnx_path(model: str, trusted: dict[str, Any]) -> Path:
    explicit = str(trusted.get("onnxPath") or trusted.get("onnx_path") or "").strip()
    candidates = [Path(explicit)] if explicit else []
    root = Path(os.environ.get("SCENEBUILDER_MODEL_ROOT", "/opt/scenebuilder-models"))
    candidates.extend([
        root / "onnx" / f"{model}.onnx",
        root / model / f"{model}.onnx",
        Path("/opt/Real-ESRGAN/weights") / f"{model}.onnx",
    ])
    for path in candidates:
        if path and path.is_file() and path.stat().st_size > 0:
            return path
    raise RuntimeError(f"TRT_BUILD_FAILED:ONNX_NOT_FOUND for {model}; provide trustedSource.onnxPath or bake ONNX into runtime")


def _engine_from_trusted(trusted: dict[str, Any], tmp: Path) -> Path:
    if trusted.get("enginePath"):
        path = Path(str(trusted["enginePath"]))
        if path.is_file():
            return path
    if trusted.get("engineKey"):
        return download_file(str(trusted["engineKey"]), tmp / "input.engine")
    raise RuntimeError("TRT_ENGINE_NOT_FOUND:trustedSource.engineKey or enginePath is required")


def _shape_args(model: str, profile: str, trusted: dict[str, Any]) -> list[str]:
    height, width = PROFILE_SHAPES.get(profile, PROFILE_SHAPES["1080 class"])
    names = trusted.get("inputNames") or trusted.get("input_names")
    if not names:
        names = ["img0", "img1"] if model == "rife-4.9" else ["input"]
    shape = f"1x3x{height}x{width}"
    return ["--shapes=" + ",".join(f"{name}:{shape}" for name in names)]


def _benchmark(binary: str, engine_path: Path, debug: list[str], profile: str) -> dict[str, Any]:
    started = time.time()
    output = _run([binary, f"--loadEngine={engine_path}", "--warmUp=500", "--duration=3"], 600, debug)
    return {"profile": profile, "durationSeconds": round(time.time() - started, 3), "summaryTail": output[-2000:]}


def _profile_shapes(model: str, profile: str, trusted: dict[str, Any]) -> dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    height, width = PROFILE_SHAPES.get(profile, PROFILE_SHAPES["1080 class"])
    names = trusted.get("inputNames") or trusted.get("input_names")
    if not names:
        names = ["img0", "img1"] if model == "rife-4.9" else ["input"]
    shape = (1, 3, height, width)
    return {str(name): (shape, shape, shape) for name in names}


def _build_with_python_api(onnx: Path, engine_path: Path, model: str, profile: str, compatibility: str, trusted: dict[str, Any], debug: list[str]) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        debug.extend(errors[-20:])
        raise RuntimeError("TRT_BUILD_FAILED:ONNX parse failed: " + " | ".join(errors[-3:]))
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    if compatibility == "AMPERE_PLUS" and hasattr(trt, "HardwareCompatibilityLevel"):
        config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.AMPERE_PLUS
    profile_obj = builder.create_optimization_profile()
    shapes = _profile_shapes(model, profile, trusted)
    for name, (minimum, optimum, maximum) in shapes.items():
        profile_obj.set_shape(name, minimum, optimum, maximum)
    config.add_optimization_profile(profile_obj)
    debug.append(f"python TensorRT builder: {trt.__version__} profile={profile} compatibility={compatibility}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT_BUILD_FAILED:build_serialized_network returned None")
    engine_path.write_bytes(bytes(serialized))


def _validate_deserialize(engine_path: Path) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("TRT_DESERIALIZE_FAILED:generated engine cannot be deserialized")


def _build_gpu_contract() -> dict[str, Any]:
    import torch

    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "multiprocessorCount": int(props.multi_processor_count),
        "computeCapability": f"{props.major}.{props.minor}",
    }


def run_engine_build(job: dict[str, Any], cancel_event, progress: Progress) -> dict[str, Any]:
    settings = job.get("settings") if isinstance(job.get("settings"), dict) else {}
    trusted = job.get("trustedSource") if isinstance(job.get("trustedSource"), dict) else {}
    trusted = {**trusted, **(settings.get("trustedSource") if isinstance(settings.get("trustedSource"), dict) else {})}
    model = _model(job)
    action = str(job.get("action") or settings.get("action") or "generate").lower()
    precision = str(job.get("precision") or settings.get("precision") or "FP16").upper()
    compatibility = str(job.get("compatibility") or settings.get("compatibility") or "AMPERE_PLUS")
    profile = str(job.get("profile") or settings.get("profile") or "1080 class")
    if precision != "FP16":
        raise RuntimeError("TRT_BUILD_FAILED:only FP16 engines are approved")
    if profile not in PROFILE_SHAPES:
        raise RuntimeError(f"TRT_BUILD_FAILED:unsupported profile {profile}")
    debug: list[str] = []
    binary = _trtexec()
    with tempfile.TemporaryDirectory(prefix="sb-trt-engine-") as raw_tmp:
        tmp = Path(raw_tmp)
        progress("preparing", 5, {"action": action, "modelFamily": model, "profile": profile})
        if action in {"validate", "benchmark"}:
            engine_path = _engine_from_trusted(trusted, tmp)
        else:
            onnx = _onnx_path(model, trusted)
            onnx_sha = _sha256(onnx)
            engine_path = tmp / f"{model}-{_slug(profile)}-{_slug(compatibility)}.engine"
            progress("building", 20, {"onnx": str(onnx), "onnxSha256": onnx_sha})
            if binary:
                args = [binary, f"--onnx={onnx}", f"--saveEngine={engine_path}", "--fp16", "--skipInference", *_shape_args(model, profile, trusted)]
                if compatibility == "AMPERE_PLUS":
                    args.append("--hardwareCompatibilityLevel=ampere+")
                _run(args, int(settings.get("buildTimeoutSeconds") or 7200), debug)
            else:
                _build_with_python_api(onnx, engine_path, model, profile, compatibility, trusted, debug)
            if cancel_event.is_set():
                raise RuntimeError("CANCELLED")
            if not engine_path.is_file() or engine_path.stat().st_size <= 0:
                raise RuntimeError("TRT_BUILD_FAILED:engine file missing after trtexec")
            trusted = {**trusted, "onnxSha256": onnx_sha}
        progress("validating", 70, {"engine": str(engine_path)})
        _validate_deserialize(engine_path)
        validation = {"ok": True, "engineSha256": _sha256(engine_path), "deserializeOk": True}
        benchmark = _benchmark(binary, engine_path, debug, profile) if binary and action in {"generate", "benchmark"} else {"skipped": "trtexec unavailable; Python TensorRT build/deserialize path used"} if action in {"generate", "benchmark"} else {}
        progress("uploading", 90, None)
        engine_sha = validation["engineSha256"]
        output = job.get("output") if isinstance(job.get("output"), dict) else {}
        output_prefix = str(settings.get("outputPrefix") or output.get("objectPrefix") or f"models/.engine/{model}/").strip("/")
        key = f"{output_prefix}/{precision.lower()}/{_slug(compatibility)}/{_slug(profile)}/{engine_sha}.engine"
        stored = upload_file(engine_path, key, "application/octet-stream") if action == "generate" else {"objectKey": trusted.get("engineKey") or key, "sizeBytes": engine_path.stat().st_size}
        return {
            **stored,
            "engineKey": stored["objectKey"],
            "engineSha256": engine_sha,
            "engineFileSizeBytes": engine_path.stat().st_size,
            "modelFamily": model,
            "precision": precision,
            "compatibility": compatibility,
            "profile": profile,
            "onnxSha256": trusted.get("onnxSha256"),
            "checkpointSha256": trusted.get("checkpointSha256"),
            "modelSourceSha256": trusted.get("modelSourceSha256"),
            "buildGpu": _build_gpu_contract(),
            "validation": validation,
            "benchmark": benchmark,
            "debug": debug[-80:],
        }

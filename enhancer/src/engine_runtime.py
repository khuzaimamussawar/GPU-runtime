from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .r2_store import download_file

_ENGINE_CACHE: dict[str, Any] = {}
_ENGINE_USE_LOGGED: set[str] = set()
_FATAL_CUDA_MARKERS = (
    "illegal memory access",
    "cudaerrorillegaladdress",
    "misaligned address",
    "launch failed",
    "unspecified launch failure",
)


def _log(event: str, **fields: Any) -> None:
    detail = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    print(f"[enhancer trt] {event}{' ' + detail if detail else ''}", flush=True)


def is_fatal_cuda_error(error: BaseException) -> bool:
    text = str(error).lower()
    return "trt_cuda_fatal" in text or any(marker in text for marker in _FATAL_CUDA_MARKERS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _engine_spec(settings: dict[str, Any], name: str) -> dict[str, Any] | None:
    engines = settings.get("engines") if isinstance(settings.get("engines"), dict) else {}
    spec = engines.get(name) or (settings.get("engine") if name == "spatial" else None)
    return spec if isinstance(spec, dict) and spec.get("engineKey") and spec.get("engineSha256") else None


def _gpu_identity(value: Any) -> str:
    return re.sub(r"(?:nvidia|geforce|generation)|[^a-z0-9]", "", str(value or "").lower())


def _engine_matches_gpu(spec: dict[str, Any]) -> bool:
    import torch

    props = torch.cuda.get_device_properties(0)
    engine_key = str(spec.get("engineKey") or "unknown")

    def unavailable(reason: str, **fields: Any) -> bool:
        log_key = f"unavailable:{reason}:{engine_key}"
        if log_key not in _ENGINE_USE_LOGGED:
            _ENGINE_USE_LOGGED.add(log_key)
            _log("engine_unavailable", reason=reason, key=engine_key, **fields)
        return False

    contract = spec.get("buildGpu") if isinstance(spec.get("buildGpu"), dict) else {}
    expected_count = int(contract.get("multiprocessorCount") or 0)
    if expected_count and expected_count != int(props.multi_processor_count):
        return unavailable("multiprocessor_count_mismatch", expected=expected_count, actual=props.multi_processor_count)
    expected_gpu = _gpu_identity(contract.get("name") or spec.get("buildGpuClass"))
    actual_gpu = _gpu_identity(props.name)
    if expected_gpu and expected_gpu != actual_gpu:
        return unavailable("gpu_model_mismatch", expected=expected_gpu, actual=actual_gpu)
    if not expected_count and not expected_gpu:
        return unavailable("missing_build_gpu_contract")
    return True


def _load_engine(spec: dict[str, Any]):
    import tensorrt as trt

    key = str(spec["engineKey"])
    expected = str(spec["engineSha256"]).lower()
    cache_key = f"{key}:{expected}"
    if cache_key in _ENGINE_CACHE:
        return _ENGINE_CACHE[cache_key]

    _log(
        "engine_download_start",
        key=key,
        profile=spec.get("profile"),
        compatibility=spec.get("compatibility"),
        sha256=expected,
    )
    with tempfile.TemporaryDirectory(prefix="sb-trt-runtime-") as tmp:
        path = download_file(key, Path(tmp) / "model.engine")
        actual = _sha256(path)
        if actual.lower() != expected:
            raise RuntimeError(f"TRT_DESERIALIZE_FAILED:engine SHA mismatch {actual} != {expected}")
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            raise RuntimeError("TRT_DESERIALIZE_FAILED:deserialize_cuda_engine returned None")
        io = []
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            io.append(f"{name}:{engine.get_tensor_dtype(name)}:{tuple(engine.get_tensor_shape(name))}")
        _log(
            "engine_ready",
            key=key,
            bytes=path.stat().st_size,
            tensors="|".join(io),
        )
    _ENGINE_CACHE[cache_key] = engine
    return engine


def _tensor_names(engine) -> list[str]:
    return [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]


def _execute(engine, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import cupy as cp
    import tensorrt as trt

    context = engine.create_execution_context()
    stream = cp.cuda.Stream(non_blocking=True)
    host_inputs: dict[str, np.ndarray] = {}
    device_inputs: dict[str, Any] = {}
    device_outputs: dict[str, Any] = {}

    try:
        # TensorRT's FP16 builder flag controls tactic/internal precision; it does
        # not guarantee FP16 network I/O. Bind buffers using each serialized
        # engine tensor's declared dtype. Keep copies on the same non-blocking
        # stream as execute_async_v3 so TRT cannot race an H2D upload.
        with stream:
            for name, array in inputs.items():
                if not engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    continue
                expected_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
                host = np.ascontiguousarray(array, dtype=expected_dtype)
                if not context.set_input_shape(name, tuple(host.shape)):
                    raise RuntimeError(f"TRT_DESERIALIZE_FAILED:input shape rejected for {name}: {tuple(host.shape)}")
                host_inputs[name] = host
                device_inputs[name] = cp.asarray(host)

            for name in _tensor_names(engine):
                mode = engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    tensor = device_inputs.get(name)
                    if tensor is None:
                        raise RuntimeError(f"TRT_DESERIALIZE_FAILED:missing input tensor {name}")
                else:
                    shape = tuple(int(dim) for dim in context.get_tensor_shape(name))
                    if not shape or any(dim <= 0 for dim in shape):
                        raise RuntimeError(f"TRT_DESERIALIZE_FAILED:invalid output shape for {name}: {shape}")
                    dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
                    tensor = cp.empty(shape, dtype=dtype)
                    device_outputs[name] = tensor
                context.set_tensor_address(name, int(tensor.data.ptr))

            ok = context.execute_async_v3(stream.ptr)
            if not ok:
                raise RuntimeError("TRT_DESERIALIZE_FAILED:execute_async_v3 failed")
        stream.synchronize()
        return {name: cp.asnumpy(tensor) for name, tensor in device_outputs.items()}
    except Exception as error:
        if is_fatal_cuda_error(error):
            raise RuntimeError(f"TRT_CUDA_FATAL:{error}") from error
        raise


def _first_output(outputs: dict[str, np.ndarray]) -> np.ndarray:
    if not outputs:
        raise RuntimeError("TRT_DESERIALIZE_FAILED:no output tensors")
    return next(iter(outputs.values()))


def try_upscale_bgr_trt(frame_bgr: np.ndarray, settings: dict[str, Any], target_w: int, target_h: int) -> np.ndarray | None:
    spec = _engine_spec(settings, "spatial")
    if not spec:
        return None
    if not _engine_matches_gpu(spec):
        return None
    key = str(spec["engineKey"])
    use_key = f"spatial:{key}"
    if use_key not in _ENGINE_USE_LOGGED:
        _ENGINE_USE_LOGGED.add(use_key)
        _log(
            "spatial_first_use",
            key=key,
            profile=spec.get("profile"),
            source=f"{frame_bgr.shape[1]}x{frame_bgr.shape[0]}",
            target=f"{target_w}x{target_h}",
        )
    engine = _load_engine(spec)
    names = _tensor_names(engine)
    input_names = [name for name in names if "input" in name.lower()] or [names[0]]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None])
    output = _first_output(_execute(engine, {input_names[0]: chw}))
    if output.ndim == 4:
        output = output[0]
    if output.shape[0] == 3:
        output = np.transpose(output, (1, 2, 0))
    output = np.clip(output.astype(np.float32), 0.0, 1.0)
    bgr = cv2.cvtColor(np.rint(output * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if bgr.shape[1] != target_w or bgr.shape[0] != target_h:
        bgr = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return bgr


def try_interpolate_rife_trt(frame0_rgb: np.ndarray, frame1_rgb: np.ndarray, timestep: float, settings: dict[str, Any]) -> np.ndarray | None:
    spec = _engine_spec(settings, "rife")
    if not spec:
        missing_key = "rife:missing_engine_spec"
        if missing_key not in _ENGINE_USE_LOGGED:
            _ENGINE_USE_LOGGED.add(missing_key)
            _log("rife_unavailable", reason="missing_engine_spec")
        return None
    if not _engine_matches_gpu(spec):
        return None
    key = str(spec["engineKey"])
    use_key = f"rife:{key}"
    if use_key not in _ENGINE_USE_LOGGED:
        _ENGINE_USE_LOGGED.add(use_key)
        _log(
            "rife_first_use",
            key=key,
            profile=spec.get("profile"),
            compatibility=spec.get("compatibility"),
            frame=f"{frame0_rgb.shape[1]}x{frame0_rgb.shape[0]}",
            timestep=round(float(timestep), 6),
        )
    engine = _load_engine(spec)
    names = _tensor_names(engine)
    lower = {name.lower(): name for name in names}
    img0_name = lower.get("img0") or lower.get("i0")
    img1_name = lower.get("img1") or lower.get("i1")
    timestep_name = lower.get("timestep") or lower.get("time") or lower.get("t")
    if not img0_name or not img1_name or not timestep_name:
        unusable_key = f"rife-unusable:{key}"
        if unusable_key not in _ENGINE_USE_LOGGED:
            _ENGINE_USE_LOGGED.add(unusable_key)
            _log("rife_unusable", reason="missing_tensor_names", key=key, tensors="|".join(names))
        return None
    left = np.ascontiguousarray(np.transpose(frame0_rgb.astype(np.float32) / 255.0, (2, 0, 1))[None])
    right = np.ascontiguousarray(np.transpose(frame1_rgb.astype(np.float32) / 255.0, (2, 0, 1))[None])
    t = np.asarray([float(timestep)], dtype=np.float32)
    output = _first_output(_execute(engine, {img0_name: left, img1_name: right, timestep_name: t}))
    if output.ndim == 4:
        output = output[0]
    if output.shape[0] == 3:
        output = np.transpose(output, (1, 2, 0))
    return np.rint(np.clip(output.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)

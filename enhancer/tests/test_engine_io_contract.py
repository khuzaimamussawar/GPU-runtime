from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_tensor_rt_inputs_follow_serialized_engine_dtype_and_stream():
    source = (ROOT / "src/engine_runtime.py").read_text(encoding="utf-8")

    assert "expected_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))" in source
    assert "host = np.ascontiguousarray(array, dtype=expected_dtype)" in source
    assert "with stream:" in source
    assert "device_inputs[name] = cp.asarray(host)" in source
    assert "context.execute_async_v3(stream.ptr)" in source
    assert "def _execution_resources(engine):" in source
    assert "_EXECUTION_CACHE[key] = resources" in source
    assert "context, stream = _execution_resources(engine)" in source

    # Callers must preserve source precision and let the serialized engine decide
    # whether each input is FP16, FP32, or another supported TensorRT dtype.
    assert "[None].astype(np.float16)" not in source


def test_tensor_rt_logs_engine_identity_and_rife_first_use():
    source = (ROOT / "src/engine_runtime.py").read_text(encoding="utf-8")
    assert '"engine_download_start"' in source
    assert '"engine_ready"' in source
    assert '"spatial_first_use"' in source
    assert '"rife_first_use"' in source
    assert '"rife_unusable"' in source
    assert 'reason="missing_tensor_names"' in source
    assert 'profile=spec.get("profile")' in source
    assert 'compatibility=spec.get("compatibility")' in source


def test_tensor_rt_refuses_engines_built_for_a_different_gpu_shape():
    source = (ROOT / "src/engine_runtime.py").read_text(encoding="utf-8")
    builder = (ROOT / "src/engine_builder.py").read_text(encoding="utf-8")

    assert "def _engine_matches_gpu" in source
    assert "multiprocessor_count_mismatch" in source
    assert "missing_build_gpu_contract" in source
    assert "same_compute_capability" in source
    assert "if not _engine_matches_gpu(spec):" in source
    assert '"buildGpu": _build_gpu_contract(compatibility)' in builder
    assert "HardwareCompatibilityLevel.SAME_COMPUTE_CAPABILITY" in builder

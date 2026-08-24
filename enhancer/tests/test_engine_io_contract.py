from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_tensor_rt_inputs_follow_serialized_engine_dtype_and_stream():
    source = (ROOT / "src/engine_runtime.py").read_text(encoding="utf-8")

    assert "expected_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))" in source
    assert "host = np.ascontiguousarray(array, dtype=expected_dtype)" in source
    assert "with stream:" in source
    assert "device_inputs[name] = cp.asarray(host)" in source
    assert "context.execute_async_v3(stream.ptr)" in source

    # Callers must preserve source precision and let the serialized engine decide
    # whether each input is FP16, FP32, or another supported TensorRT dtype.
    assert "[None].astype(np.float16)" not in source

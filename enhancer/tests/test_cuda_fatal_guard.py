from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_engine_runtime_classifies_illegal_address_as_fatal():
    source = _read("src/engine_runtime.py")
    assert '"illegal memory access"' in source
    assert '"cudaerrorillegaladdress"' in source
    assert 'raise RuntimeError(f"TRT_CUDA_FATAL:{error}")' in source
    assert "context.set_input_shape" in source
    assert "invalid output shape" in source


def test_fast_and_quality_vfi_never_native_fallback_after_fatal_cuda():
    fast = _read("src/video_pipeline.py")
    quality_vfi = _read("src/vfi_postprocess.py")
    assert 'if is_fatal_cuda_error(error) or not settings.get("allowNativeFallback", True): raise' in fast
    assert 'if is_fatal_cuda_error(error) or (settings and not settings.get("allowNativeFallback", True)):' in quality_vfi


def test_runtime_entrypoint_quarantines_poisoned_cuda_context():
    guard = _read("src/server_guard.py")
    fast_docker = _read("docker/Dockerfile.fast")
    quality_docker = _read("docker/Dockerfile.quality")
    assert 'return "CUDA_ILLEGAL_ADDRESS"' in guard
    assert 'server._READY = False' in guard
    assert 'server._DRAINING = True' in guard
    assert '"worker_unhealthy"' in guard
    assert 'CMD ["python3", "-m", "src.server_guard"]' in fast_docker
    assert 'CMD ["python3", "-m", "src.server_guard"]' in quality_docker

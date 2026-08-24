from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "src" / "models.py").read_text(encoding="utf-8")


def test_rife_loader_adds_repository_root_for_model_package():
    assert 'RIFE_ROOT = Path(os.environ.get("RIFE_ROOT", "/opt/Practical-RIFE"))' in SOURCE
    assert 'import_roots = [RIFE_ROOT, module_path.parent]' in SOURCE
    assert 'if module_path.parent.name == "train_log":' in SOURCE
    assert 'import_roots.append(module_path.parent.parent)' in SOURCE
    assert 'sys.path.insert(0, value)' in SOURCE


def test_rife_loader_exposes_flat_snapshot_as_train_log_package():
    assert 'def _install_rife_train_log_alias(module_path: Path)' in SOURCE
    assert 'if module_path.parent != RIFE_MODEL_DIR:' in SOURCE
    assert 'ifnet_path = RIFE_MODEL_DIR / "IFNet_HDv3.py"' in SOURCE
    assert 'package = types.ModuleType("train_log")' in SOURCE
    assert 'package.__path__ = [str(RIFE_MODEL_DIR)]' in SOURCE
    assert 'sys.modules["train_log"] = package' in SOURCE
    assert '_install_rife_train_log_alias(module_path)' in SOURCE


def test_native_rife_fallback_announces_model_loading():
    assert '[enhancer rife] native_load_start' in SOURCE
    assert '[enhancer rife] native_load_ready' in SOURCE


def test_native_rife_pads_4k_height_to_ifnet_alignment_and_crops_result():
    assert 'def _rife_padded_shape(width: int, height: int, scale: float)' in SOURCE
    assert 'block = max(32, int(32 / safe_scale))' in SOURCE
    assert 'i0 = F.pad(i0, padding)' in SOURCE
    assert 'i1 = F.pad(i1, padding)' in SOURCE
    assert 'result = result[..., :height, :width]' in SOURCE
    assert '[enhancer rife] native_pad' in SOURCE

    namespace = {}
    exec(compile('\n'.join([
        'def shape(width, height, scale):',
        '    safe_scale = max(float(scale), 1e-6)',
        '    block = max(32, int(32 / safe_scale))',
        '    padded_w = ((int(width) - 1) // block + 1) * block',
        '    padded_h = ((int(height) - 1) // block + 1) * block',
        '    return padded_w, padded_h',
    ]), '<rife-shape-test>', 'exec'), namespace)
    assert namespace['shape'](3840, 2160, 1.0) == (3840, 2176)
    assert namespace['shape'](1920, 1080, 1.0) == (1920, 1088)

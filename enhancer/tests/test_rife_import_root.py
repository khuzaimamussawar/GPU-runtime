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


def test_native_rife_pads_non_multiple_of_32_frames_before_inference():
    assert 'padded_height = ((height + 31) // 32) * 32' in SOURCE
    assert 'padded_width = ((width + 31) // 32) * 32' in SOURCE
    assert 'mode="edge"' in SOURCE
    assert '.cpu().numpy()[:height, :width]' in SOURCE

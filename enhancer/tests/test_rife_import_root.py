from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "src" / "models.py").read_text(encoding="utf-8")


def test_rife_loader_adds_repository_root_for_model_package():
    assert 'RIFE_ROOT = Path(os.environ.get("RIFE_ROOT", "/opt/Practical-RIFE"))' in SOURCE
    assert 'import_roots = [RIFE_ROOT, module_path.parent]' in SOURCE
    assert 'if module_path.parent.name == "train_log":' in SOURCE
    assert 'import_roots.append(module_path.parent.parent)' in SOURCE
    assert 'sys.path.insert(0, value)' in SOURCE

import json
from pathlib import Path

from src.common.job_contract import H3Job, normalize_fps
from src.common.runtime_adapter import _round_h3_frames

assert normalize_fps(24) == 24
assert normalize_fps(30) == 30
assert normalize_fps("30") == 30
assert normalize_fps(60) == 24

job24 = H3Job("24", "p", "h3_fl2va", "t2v", "", 1056, 594, 5.0, fps=24)
job30 = H3Job("30", "p", "h3_fl2va", "t2v", "", 1056, 594, 5.0, fps=30)
assert _round_h3_frames(job24) == 124
assert _round_h3_frames(job30) == 158
assert _round_h3_frames(H3Job("x", "p", "h3_fl2va", "t2v", "", 1, 1, 4.688, fps=24)) == 124
assert _round_h3_frames(H3Job("y", "p", "h3_fl2va", "t2v", "", 1, 1, 4.688, fps=30)) == 141

fl_manifest = json.loads(Path("workflows/manifests/fl2va_manifest.json").read_text())
ref_manifest = json.loads(Path("workflows/manifests/ref2va_manifest.json").read_text())
assert fl_manifest["requiredPaths"]["fps"] == "21.inputs.fps"
assert ref_manifest["requiredPaths"]["fps"] == "15.inputs.fps"

fl_workflow = json.loads(Path("workflows/fl2va_master.json").read_text())
ref_workflow = json.loads(Path("workflows/ref2va_master.json").read_text())
assert fl_workflow["21"]["inputs"]["fps"] == 24
assert ref_workflow["15"]["inputs"]["fps"] == 24

for handler in (
    Path("src/runpod/fl2va_handler.py"),
    Path("src/runpod/ref2va_handler.py"),
    Path("src/novita/fl2va_handler.py"),
    Path("src/novita/ref2va_handler.py"),
    Path("src/pod/server.py"),
):
    text = handler.read_text()
    assert "runtime_adapter import run_h3_job" in text, handler
    assert "run_h3_job(" in text, handler

print("H3 FPS runtime contract: PASS")

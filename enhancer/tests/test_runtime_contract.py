import json
import os
from pathlib import Path
from unittest.mock import patch

from enhancer.src import callbacks, video_encoder
from enhancer.src.config import RuntimeConfig


ROOT = Path(__file__).parents[1]
CONTRACT = json.loads((ROOT / "runtime-contract.json").read_text())


def test_shared_transport_and_runtime_defaults_match_contract():
    values = {
        "SCENEBUILDER_WORKER_ID": "worker-test",
        "SCENEBUILDER_POD_TOKEN": "token-test",
        "SCENEBUILDER_CONTROL_URL": "https://example.invalid",
        "R2_BUCKET_NAME": "bucket",
        "R2_ENDPOINT": "https://r2.invalid",
        "R2_ACCESS_KEY": "access",
        "R2_SECRET_KEY": "secret",
    }
    with patch.dict(os.environ, values, clear=True):
        cfg = RuntimeConfig.from_env()
    assert cfg.port == CONTRACT["podPort"] == 8000
    assert cfg.idle_timeout_seconds == CONTRACT["idleTimeoutSeconds"] == 60
    assert callbacks.H3_EVENT_PATH == CONTRACT["callbackPath"]


def test_encoder_contract_matches_scene_control_plane():
    encoder = CONTRACT["videoEncoder"]
    assert video_encoder.QUALITY_MIN == encoder["qualityMin"] == 12
    assert video_encoder.QUALITY_MAX == encoder["qualityMax"] == 25
    assert video_encoder.normalize_video_encoder({}) == encoder["default"] == "nvenc"

    # Availability is tested separately; this assertion pins the public codec names
    # and default quality controls that SceneBuilder persists into every video job.
    assert encoder["nvenc"] == {
        "codec": "hevc_nvenc",
        "qualityField": "nvencCq",
        "defaultQuality": 17,
    }
    assert encoder["x265"] == {
        "codec": "libx265",
        "qualityField": "x265Crf",
        "defaultQuality": 15,
    }


def test_service_kind_contract_contains_all_gpu_runtime_modes():
    assert set(CONTRACT["serviceKinds"]) == {
        "enhancer_fast",
        "enhancer_quality",
        "enhancer_engine_builder",
    }


def test_container_logs_show_job_stage_engine_and_resource_context():
    source = (ROOT / "src/server.py").read_text(encoding="utf-8")
    assert "def _resource_snapshot()" in source
    assert '"diskTotalGiB"' in source
    assert '"diskUsedGiB"' in source
    assert '"diskFreeGiB"' in source
    assert '"gpuAllocatedMiB"' in source
    assert '"job_start"' in source
    assert '"job_progress"' in source
    assert '"job_completed"' in source
    assert '"job_failed"' in source
    assert '"cancel_requested"' in source
    assert "progress_value >= record.last_log_progress + 10.0" in source
    assert "engines=_engine_log_summary(settings)" in source


def test_final_images_request_nvidia_video_driver_capability_for_nvenc():
    for name in ("Dockerfile.fast", "Dockerfile.quality"):
        dockerfile = (ROOT / "docker" / name).read_text(encoding="utf-8")
        assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video" in dockerfile
        assert "hevc_nvenc" in dockerfile
        assert "libx265" in dockerfile

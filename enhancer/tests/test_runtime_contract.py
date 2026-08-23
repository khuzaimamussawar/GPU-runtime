import json
import os
from pathlib import Path
from unittest.mock import patch

from enhancer.src import callbacks, video_encoder
from enhancer.src.config import RuntimeConfig


CONTRACT = json.loads((Path(__file__).parents[1] / "runtime-contract.json").read_text())


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

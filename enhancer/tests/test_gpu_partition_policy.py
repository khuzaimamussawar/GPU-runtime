from unittest.mock import patch

from enhancer.src import gpu


def test_partitioned_gpu_is_reported_but_not_rejected():
    with patch.object(gpu, "_cmd") as command:
        command.side_effect = [
            "NVIDIA A100 MIG 1g.10gb, 580.65.06, 10240, 8.0",
            "MIG Mode\n    Current : Enabled\n",
        ]
        details = gpu._nvidia_query()

    assert details["partitioned"] is True
    assert details["name"] == "NVIDIA A100 MIG 1g.10gb"

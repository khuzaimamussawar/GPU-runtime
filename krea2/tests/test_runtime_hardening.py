from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.image_pod import media, server
from src.image_pod.adapters.krea2 import Krea2Adapter


class _FakeHandler:
    def __init__(self, length: int, body: bytes = b"") -> None:
        self.headers = {"Content-Length": str(length)}
        self.rfile = io.BytesIO(body)


class ImagePodRuntimeHardeningTests(unittest.TestCase):
    def test_missing_pod_token_never_authorizes(self):
        fake = _FakeHandler(0)
        fake.headers["Authorization"] = "Bearer anything"
        with mock.patch.object(server, "POD_TOKEN", ""):
            self.assertFalse(server.Handler._authorized(fake))

    def test_pod_auth_uses_exact_bearer_token(self):
        with mock.patch.object(server, "POD_TOKEN", "secret"):
            good = _FakeHandler(0)
            good.headers["Authorization"] = "Bearer secret"
            bad = _FakeHandler(0)
            bad.headers["Authorization"] = "Bearer secret2"
            self.assertTrue(server.Handler._authorized(good))
            self.assertFalse(server.Handler._authorized(bad))

    def test_job_body_over_one_mib_is_rejected_before_read(self):
        fake = _FakeHandler(server.REQUEST_MAX_BYTES + 1)
        with self.assertRaises(server.RequestTooLargeError):
            server.Handler._read_json(fake)

    def test_fatal_cuda_and_comfy_errors_are_classified(self):
        self.assertTrue(server._is_fatal_runtime_error(RuntimeError("CUDA out of memory")))
        self.assertTrue(server._is_fatal_runtime_error(RuntimeError("ComfyUI exited during startup with code 1")))
        self.assertFalse(server._is_fatal_runtime_error(ValueError("invalid request")))

    def test_runtime_configuration_requires_auth_control_and_r2(self):
        with mock.patch.object(server, "POD_TOKEN", "token"), \
             mock.patch.object(server, "WORKER_ID", "worker"), \
             mock.patch.object(server, "CONTROL_URL", "https://control.invalid/events"), \
             mock.patch.dict(os.environ, {
                 "R2_BUCKET_NAME": "bucket",
                 "R2_ENDPOINT": "https://r2.invalid",
                 "R2_ACCESS_KEY": "key",
                 "R2_SECRET_KEY": "secret",
             }, clear=False):
            self.assertEqual(server._required_runtime_config_errors(), [])

    def test_full_output_upload_fails_closed_without_r2(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            path.write_bytes(b"png")
            with mock.patch.object(media, "_r2_client", return_value=(None, None)):
                with self.assertRaises(media.ImageMediaError):
                    media._upload_file(path, "projects/p/image.png", "image/png")

    def test_arbitrary_external_reference_url_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "ref.png"
            with self.assertRaises(media.ImageMediaError):
                media._download_ref_to_path(
                    "https://evil.example/random.png",
                    target,
                    allowed_prefixes=media.STYLE_REFERENCE_PREFIXES,
                    max_bytes=1024,
                )

    def test_lora_requires_trusted_size_sha_and_strength_bounds(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(media, "LORA_CACHE", Path(tmp)):
            base = {
                "loraId": "lora-1",
                "objectKey": "models/lora/krea2/lora-1.safetensors",
                "strength": 0.5,
                "minStrength": 0.0,
                "maxStrength": 1.0,
            }
            with self.assertRaises(media.ImageMediaError):
                media.materialize_user_loras([base])

            with_size = {**base, "fileSizeBytes": 123}
            with self.assertRaises(media.ImageMediaError):
                media.materialize_user_loras([with_size])

    def test_krea_worker_rejects_random_mode_without_resolved_seed(self):
        adapter = Krea2Adapter()
        with self.assertRaisesRegex(ValueError, "SceneBuilder must resolve random seed"):
            adapter._normalize_settings({
                "settings": {
                    "seedMode": "random",
                    "seed": None,
                }
            })

    def test_krea_worker_preserves_fixed_seed(self):
        adapter = Krea2Adapter()
        settings = adapter._normalize_settings({
            "settings": {
                "seedMode": "fixed",
                "seed": 424242,
            }
        })
        self.assertEqual(settings["seedMode"], "fixed")
        self.assertEqual(settings["seed"], 424242)

    def test_krea_worker_preserves_scene_resolved_random_seed(self):
        adapter = Krea2Adapter()
        settings = adapter._normalize_settings({
            "settings": {
                "seedMode": "random",
                "seed": 987654321,
            }
        })
        self.assertEqual(settings["seedMode"], "random")
        self.assertEqual(settings["seed"], 987654321)

    def test_krea_worker_preserves_legacy_explicit_seed_as_fixed(self):
        adapter = Krea2Adapter()
        settings = adapter._normalize_settings({"settings": {"seed": 77}})
        self.assertEqual(settings["seedMode"], "fixed")
        self.assertEqual(settings["seed"], 77)

    def test_krea_worker_rejects_fixed_seed_without_value(self):
        adapter = Krea2Adapter()
        with self.assertRaisesRegex(ValueError, "seed is required"):
            adapter._normalize_settings({"settings": {"seedMode": "fixed", "seed": None}})

    def test_krea_worker_rejects_missing_seed_without_mode(self):
        adapter = Krea2Adapter()
        with self.assertRaisesRegex(ValueError, "seed is required"):
            adapter._normalize_settings({"settings": {}})

    def test_idle_renew_extends_idle_and_recent_idle_expiry_only(self):
        state = server.ImagePodState()
        state.worker_status = "idle"
        state.current_job_id = None
        state.draining = False
        state.idle_timeout_seconds = 90

        with mock.patch.object(server.time, "time", return_value=1000.0):
            renewed = state.renew_idle()
        self.assertIsNotNone(renewed)
        self.assertEqual(renewed["status"], "idle")
        self.assertEqual(renewed["idleSince"], 1000.0)
        self.assertEqual(renewed["terminateAfter"], 1090.0)

        state.worker_status = "busy"
        state.current_job_id = "job-1"
        self.assertIsNone(state.renew_idle())

        state.worker_status = "draining"
        state.current_job_id = None
        state.draining = True
        state.terminate_after = 995.0
        with mock.patch.object(server.time, "time", return_value=1000.0):
            renewed = state.renew_idle()
        self.assertIsNotNone(renewed)
        self.assertFalse(state.draining)
        self.assertEqual(state.worker_status, "idle")

        state.worker_status = "draining"
        state.current_job_id = None
        state.draining = True
        state.terminate_after = 900.0
        with mock.patch.object(server.time, "time", return_value=1000.0):
            self.assertIsNone(state.renew_idle())

        state.worker_status = "unhealthy"
        state.current_job_id = None
        state.draining = True
        state.terminate_after = 995.0
        with mock.patch.object(server.time, "time", return_value=1000.0):
            self.assertIsNone(state.renew_idle())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import io
import inspect
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from src.image_pod import media, server
from src.image_pod.adapters.krea2 import Krea2Adapter, _verify_lora_graph


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

    def test_generated_output_is_not_uploaded_by_the_pod(self):
        source = inspect.getsource(media.finalize_image_outputs)
        self.assertIn("SceneBuilder, not the GPU pod, stores generated output in R2.", source)
        self.assertNotIn("_upload_file", source)
        self.assertNotIn("outputImagePrefix", source)

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

    def test_baked_lora_resolves_from_model_root_without_r2_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "snorvdoc.safetensors"
            target.write_bytes(b"baked lora bytes")
            digest = media._hash_file(target)
            with mock.patch.object(media, "KREA2_BAKED_LORA_DIR", root):
                resolved = media.materialize_user_loras([
                    {
                        "loraId": "snorvdoc",
                        "storageSource": "baked",
                        "fileName": "snorvdoc.safetensors",
                        "bakedPath": str(target),
                        "fileSizeBytes": target.stat().st_size,
                        "sha256": digest,
                        "strength": 2.0,
                        "minStrength": 0.0,
                        "maxStrength": 3.0,
                    }
                ])

        self.assertEqual(resolved[0]["fileName"], "snorvdoc.safetensors")

    def test_krea_lora_graph_audit_verifies_two_lora_chain_and_logs_identity(self):
        manifest = {
            "runtimeGraphPatching": {
                "userLoras": {
                    "baseModelNode": "30:10",
                    "consumerNode": "30:3",
                    "consumerInput": "model",
                    "loaderClass": "LoraLoaderModelOnly",
                    "nameInput": "lora_name",
                    "strengthInput": "strength_model",
                }
            }
        }
        loras = [
            {
                "loraId": "krea2-darkchurch-style",
                "fileName": "krea2-darkchurch-style.safetensors",
                "strength": 1.11,
            },
            {
                "loraId": "krea2-minimalistic-vector-art",
                "fileName": "krea2-minimalistic-vector-art.safetensors",
                "strength": 0.22,
            },
        ]
        workflow = {
            "30:10": {"class_type": "UNETLoader", "inputs": {}},
            "sb_lora_01": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "lora_name": "krea2-darkchurch-style.safetensors",
                    "strength_model": 1.11,
                    "model": ["30:10", 0],
                },
            },
            "sb_lora_02": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "lora_name": "krea2-minimalistic-vector-art.safetensors",
                    "strength_model": 0.22,
                    "model": ["sb_lora_01", 0],
                },
            },
            "30:3": {"class_type": "KSampler", "inputs": {"model": ["sb_lora_02", 0]}},
        }

        output = io.StringIO()
        with redirect_stdout(output):
            loaded = _verify_lora_graph("job-1", workflow, manifest, loras)

        self.assertEqual([item["loraId"] for item in loaded], [
            "krea2-darkchurch-style",
            "krea2-minimalistic-vector-art",
        ])
        self.assertEqual(loaded[0]["strength"], 1.11)
        self.assertEqual(loaded[1]["strength"], 0.22)
        logs = output.getvalue()
        self.assertIn("id=krea2-darkchurch-style", logs)
        self.assertIn("id=krea2-minimalistic-vector-art", logs)
        self.assertIn("graph verified count=2", logs)

    def test_krea_lora_graph_audit_fails_closed_when_second_lora_is_missing(self):
        manifest = {
            "runtimeGraphPatching": {
                "userLoras": {
                    "baseModelNode": "30:10",
                    "consumerNode": "30:3",
                    "consumerInput": "model",
                    "loaderClass": "LoraLoaderModelOnly",
                    "nameInput": "lora_name",
                    "strengthInput": "strength_model",
                }
            }
        }
        loras = [
            {"loraId": "one", "fileName": "one.safetensors", "strength": 1.0},
            {"loraId": "two", "fileName": "two.safetensors", "strength": 0.5},
        ]
        workflow = {
            "30:10": {"class_type": "UNETLoader", "inputs": {}},
            "sb_lora_01": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "lora_name": "one.safetensors",
                    "strength_model": 1.0,
                    "model": ["30:10", 0],
                },
            },
            "30:3": {"class_type": "KSampler", "inputs": {"model": ["sb_lora_01", 0]}},
        }
        with self.assertRaisesRegex(RuntimeError, "missing expected node sb_lora_02"):
            _verify_lora_graph("job-2", workflow, manifest, loras)

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

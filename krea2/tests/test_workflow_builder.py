from __future__ import annotations

import json
import unittest
from pathlib import Path

from krea2.runtime.workflow_builder import (
    MAX_SAFE_SEED,
    WorkflowBuildError,
    prepare_krea2_workflow,
)


ROOT = Path(__file__).resolve().parents[2]
# In the source checkout workflows live under krea2/workflows. In the final
# runtime image the workflow layer installs them at /opt/scenebuilder-image/workflows.
# Keep the same test suite valid in both layouts so the runtime Docker build can
# execute it without duplicating workflow assets into the final image.
_SOURCE_WORKFLOW_ROOT = ROOT / "krea2" / "workflows"
_INSTALLED_WORKFLOW_ROOT = ROOT / "workflows"
WORKFLOW_ROOT = _SOURCE_WORKFLOW_ROOT if _SOURCE_WORKFLOW_ROOT.is_dir() else _INSTALLED_WORKFLOW_ROOT


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def lora(name: str, strength: float = 1.0):
    return {
        "loraId": name,
        "fileName": f"{name}.safetensors",
        "strength": strength,
        "minStrength": 0.0,
        "maxStrength": 1.0,
    }


class Krea2WorkflowBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t2i = load_json(WORKFLOW_ROOT / "krea2_turbo.json")
        cls.t2i_manifest = load_json(WORKFLOW_ROOT / "manifests" / "krea2_turbo.json")
        cls.style = load_json(WORKFLOW_ROOT / "krea2_style_reference.json")
        cls.style_manifest = load_json(WORKFLOW_ROOT / "manifests" / "krea2_style_reference.json")

    def test_t2i_zero_lora_uses_base_model(self):
        prepared = prepare_krea2_workflow(
            self.t2i,
            self.t2i_manifest,
            settings={"width": 1280, "height": 720, "seed": 7},
            user_loras=[],
        )
        self.assertEqual(prepared["30:3"]["inputs"]["model"], ["30:10", 0])
        self.assertEqual(prepared["30:3"]["inputs"]["seed"], 7)
        self.assertNotIn("sb_lora_01", prepared)

    def test_t2i_three_loras_form_ordered_chain(self):
        loras = [lora("a", 0.25), lora("b", 0.50), lora("c", 0.75)]
        prepared = prepare_krea2_workflow(self.t2i, self.t2i_manifest, user_loras=loras)
        self.assertEqual(prepared["sb_lora_01"]["inputs"]["model"], ["30:10", 0])
        self.assertEqual(prepared["sb_lora_02"]["inputs"]["model"], ["sb_lora_01", 0])
        self.assertEqual(prepared["sb_lora_03"]["inputs"]["model"], ["sb_lora_02", 0])
        self.assertEqual(prepared["30:3"]["inputs"]["model"], ["sb_lora_03", 0])

    def test_lora_strength_outside_catalog_range_is_rejected(self):
        with self.assertRaisesRegex(WorkflowBuildError, "INVALID_LORA_STRENGTH"):
            prepare_krea2_workflow(
                self.t2i,
                self.t2i_manifest,
                user_loras=[lora("bad", 1.5)],
            )

    def test_lora_requires_catalog_range(self):
        with self.assertRaises(WorkflowBuildError):
            prepare_krea2_workflow(
                self.t2i,
                self.t2i_manifest,
                user_loras=[{"loraId": "x", "fileName": "x.safetensors", "strength": 0.5}],
            )

    def test_t2i_four_loras_is_rejected(self):
        with self.assertRaises(WorkflowBuildError):
            prepare_krea2_workflow(
                self.t2i,
                self.t2i_manifest,
                user_loras=[lora(str(i)) for i in range(4)],
            )

    def test_only_approved_render_sizes_are_allowed(self):
        for width, height in ((1280, 720), (720, 1280), (2048, 1152), (1152, 2048)):
            prepared = prepare_krea2_workflow(
                self.t2i,
                self.t2i_manifest,
                settings={"width": width, "height": height},
            )
            self.assertEqual(prepared["30:5"]["inputs"]["width"], width)
            self.assertEqual(prepared["30:5"]["inputs"]["height"], height)

        with self.assertRaises(WorkflowBuildError):
            prepare_krea2_workflow(
                self.t2i,
                self.t2i_manifest,
                settings={"width": 1920, "height": 1080},
            )
        with self.assertRaises(WorkflowBuildError):
            prepare_krea2_workflow(self.t2i, self.t2i_manifest, settings={"width": 1280})

    def test_sampler_scheduler_seed_and_cfg_limits(self):
        with self.assertRaises(WorkflowBuildError):
            prepare_krea2_workflow(self.t2i, self.t2i_manifest, settings={"sampler": "random_sampler"})
        with self.assertRaises(WorkflowBuildError):
            prepare_krea2_workflow(self.t2i, self.t2i_manifest, settings={"scheduler": "random_scheduler"})
        with self.assertRaises(WorkflowBuildError):
            prepare_krea2_workflow(self.t2i, self.t2i_manifest, settings={"seed": MAX_SAFE_SEED + 1})
        with self.assertRaises(WorkflowBuildError):
            prepare_krea2_workflow(self.t2i, self.t2i_manifest, settings={"cfg": 0.99})
        with self.assertRaises(WorkflowBuildError):
            prepare_krea2_workflow(self.t2i, self.t2i_manifest, settings={"cfg": 1.51})

    def test_krea_steps_are_limited_to_product_range(self):
        for steps in (4, 8, 12):
            prepared = prepare_krea2_workflow(self.t2i, self.t2i_manifest, settings={"steps": steps})
            self.assertEqual(prepared["30:3"]["inputs"]["steps"], steps)
        for steps in (3, 13):
            with self.assertRaises(WorkflowBuildError):
                prepare_krea2_workflow(self.t2i, self.t2i_manifest, settings={"steps": steps})

    def test_prompt_enhancement_nodes_are_absent_for_krea(self):
        enhancer_nodes = {"30:16", "30:17", "30:18", "30:20", "30:21", "30:24"}
        self.assertTrue(enhancer_nodes.isdisjoint(self.t2i))
        self.assertTrue(enhancer_nodes.isdisjoint(self.style))
        self.assertNotIn("promptEnhance", self.t2i_manifest.get("requiredPaths", {}))
        self.assertNotIn("promptEnhance", self.style_manifest.get("requiredPaths", {}))

        t2i = prepare_krea2_workflow(
            self.t2i,
            self.t2i_manifest,
            settings={"promptEnhance": True},
        )
        self.assertEqual(t2i["30:6"]["inputs"]["text"], ["30:19", 0])

        style = prepare_krea2_workflow(
            self.style,
            self.style_manifest,
            settings={"promptEnhance": True},
            reference_images=["style.png"],
        )
        self.assertEqual(style["30:52"]["inputs"]["prompt"], ["30:19", 0])

    def test_negative_prompt_is_guided_only_for_t2i(self):
        native = prepare_krea2_workflow(
            self.t2i,
            self.t2i_manifest,
            settings={"cfg": 1.0, "negativePrompt": "blurry"},
        )
        self.assertNotIn("sb_negative", native)
        self.assertEqual(native["30:3"]["inputs"]["negative"], ["30:13", 0])

        guided = prepare_krea2_workflow(
            self.t2i,
            self.t2i_manifest,
            settings={"cfg": 1.2, "negativePrompt": "blurry"},
        )
        self.assertEqual(guided["sb_negative"]["class_type"], "CLIPTextEncode")
        self.assertEqual(guided["sb_negative"]["inputs"]["text"], "blurry")
        self.assertEqual(guided["sb_negative"]["inputs"]["clip"], ["30:11", 0])
        self.assertEqual(guided["30:3"]["inputs"]["negative"], ["sb_negative", 0])

    def test_style_supports_ten_reference_images(self):
        refs = [f"style-{i}.png" for i in range(1, 11)]
        prepared = prepare_krea2_workflow(
            self.style,
            self.style_manifest,
            settings={
                "width": 1152,
                "height": 2048,
                "seed": 123,
                "steps": 8,
                "cfg": 1.0,
                "sampler": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
            reference_images=refs,
        )
        encoder_inputs = prepared["30:52"]["inputs"]
        self.assertEqual(prepared["69"]["inputs"]["image"], "style-1.png")
        self.assertEqual(encoder_inputs["image1"], ["69", 0])
        for index in range(2, 11):
            node_id = f"sb_ref_{index:02d}"
            self.assertEqual(prepared[node_id]["inputs"]["image"], f"style-{index}.png")
            self.assertEqual(encoder_inputs[f"image{index}"], [node_id, 0])
        self.assertEqual(prepared["30:61"]["inputs"]["width"], 1152)
        self.assertEqual(prepared["30:64"]["inputs"]["height"], 2048)
        self.assertEqual(prepared["30:63"]["inputs"]["noise_seed"], 123)
        self.assertEqual(prepared["30:64"]["inputs"]["model"], ["30:15", 0])

    def test_style_requires_one_to_ten_refs(self):
        with self.assertRaises(WorkflowBuildError):
            prepare_krea2_workflow(self.style, self.style_manifest, reference_images=[])
        with self.assertRaises(WorkflowBuildError):
            prepare_krea2_workflow(
                self.style,
                self.style_manifest,
                reference_images=[f"{i}.png" for i in range(11)],
            )

    def test_style_combines_refs_with_three_user_loras_after_system_adapter(self):
        loras = [lora("a", 0.25), lora("b", 0.50), lora("c", 0.75)]
        prepared = prepare_krea2_workflow(
            self.style,
            self.style_manifest,
            user_loras=loras,
            reference_images=["style-1.png", "style-2.png"],
        )
        self.assertEqual(prepared["30:15"]["inputs"]["model"], ["30:10", 0])
        self.assertEqual(prepared["sb_lora_01"]["inputs"]["model"], ["30:15", 0])
        self.assertEqual(prepared["sb_lora_02"]["inputs"]["model"], ["sb_lora_01", 0])
        self.assertEqual(prepared["sb_lora_03"]["inputs"]["model"], ["sb_lora_02", 0])
        self.assertEqual(prepared["30:64"]["inputs"]["model"], ["sb_lora_03", 0])
        self.assertEqual(prepared["30:52"]["inputs"]["image1"], ["69", 0])
        self.assertEqual(prepared["30:52"]["inputs"]["image2"], ["sb_ref_02", 0])

    def test_style_guided_negative_prompt(self):
        guided = prepare_krea2_workflow(
            self.style,
            self.style_manifest,
            settings={"cfg": 1.2, "negativePrompt": "text artifacts"},
            reference_images=["style.png"],
        )
        self.assertEqual(guided["30:57"]["inputs"]["negative"], ["sb_negative", 0])
        self.assertEqual(guided["sb_negative"]["inputs"]["clip"], ["30:11", 0])

    def test_style_four_user_loras_is_rejected(self):
        with self.assertRaises(WorkflowBuildError):
            prepare_krea2_workflow(
                self.style,
                self.style_manifest,
                user_loras=[lora(str(i)) for i in range(4)],
                reference_images=["style.png"],
            )

    def test_runtime_vae_selection(self):
        prepared = prepare_krea2_workflow(
            self.t2i,
            self.t2i_manifest,
            settings={"vae": "wan_2_1"},
        )
        self.assertEqual(prepared["30:12"]["inputs"]["vae_name"], "wan_2.1_vae.safetensors")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import unittest
from pathlib import Path

from krea2.runtime.workflow_builder import WorkflowBuildError, prepare_krea2_workflow


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = ROOT / "krea2" / "workflows"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
        loras = [
            {"loraId": "a", "fileName": "a.safetensors", "strength": 0.25, "minStrength": 0.0, "maxStrength": 1.0},
            {"loraId": "b", "fileName": "b.safetensors", "strength": 0.50, "minStrength": 0.0, "maxStrength": 1.0},
            {"loraId": "c", "fileName": "c.safetensors", "strength": 0.75, "minStrength": 0.0, "maxStrength": 1.0},
        ]
        prepared = prepare_krea2_workflow(self.t2i, self.t2i_manifest, user_loras=loras)
        self.assertEqual(prepared["sb_lora_01"]["inputs"]["model"], ["30:10", 0])
        self.assertEqual(prepared["sb_lora_02"]["inputs"]["model"], ["sb_lora_01", 0])
        self.assertEqual(prepared["sb_lora_03"]["inputs"]["model"], ["sb_lora_02", 0])
        self.assertEqual(prepared["30:3"]["inputs"]["model"], ["sb_lora_03", 0])

    def test_lora_strength_outside_catalog_range_is_rejected(self):
        with self.assertRaises(WorkflowBuildError):
            prepare_krea2_workflow(
                self.t2i,
                self.t2i_manifest,
                user_loras=[
                    {
                        "loraId": "bad",
                        "fileName": "bad.safetensors",
                        "strength": 1.5,
                        "minStrength": 0.0,
                        "maxStrength": 1.0,
                    }
                ],
            )

    def test_t2i_four_loras_is_rejected(self):
        with self.assertRaises(WorkflowBuildError):
            prepare_krea2_workflow(
                self.t2i,
                self.t2i_manifest,
                user_loras=[
                    {"fileName": f"{i}.safetensors", "strength": 1.0}
                    for i in range(4)
                ],
            )

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

    def test_style_requires_one_to_ten_refs(self):
        with self.assertRaises(WorkflowBuildError):
            prepare_krea2_workflow(self.style, self.style_manifest, reference_images=[])
        with self.assertRaises(WorkflowBuildError):
            prepare_krea2_workflow(
                self.style,
                self.style_manifest,
                reference_images=[f"{i}.png" for i in range(11)],
            )

    def test_style_rejects_user_lora(self):
        with self.assertRaises(WorkflowBuildError):
            prepare_krea2_workflow(
                self.style,
                self.style_manifest,
                user_loras=[{"fileName": "x.safetensors", "strength": 1.0}],
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

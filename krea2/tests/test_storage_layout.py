from __future__ import annotations

import inspect
import unittest

from src.image_pod import media


class Krea2StorageLayoutTests(unittest.TestCase):
    def test_runtime_returns_local_artifacts_for_scenebuilder_to_store(self):
        source = inspect.getsource(media.finalize_image_outputs)
        self.assertIn('final_dir = IMAGE_ROOT / "tmp" / safe_name(job_id)', source)
        self.assertIn('"fileName": final_path.name', source)
        self.assertIn('"fileName": thumb_path.name', source)
        self.assertIn('SceneBuilder, not the GPU pod, stores generated output in R2.', source)

    def test_runtime_never_uses_r2_output_prefixes_for_generated_media(self):
        source = inspect.getsource(media.finalize_image_outputs)
        self.assertNotIn('outputImagePrefix', source)
        self.assertNotIn('outputThumbnailPrefix', source)
        self.assertNotIn('_upload_file', source)


if __name__ == "__main__":
    unittest.main()

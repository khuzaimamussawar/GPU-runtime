from __future__ import annotations

import inspect
import unittest

from src.image_pod import media


class Krea2StorageLayoutTests(unittest.TestCase):
    def test_runtime_uses_separate_canonical_image_and_thumbnail_prefixes(self):
        source = inspect.getsource(media.finalize_image_outputs)
        self.assertIn('settings.get("outputImagePrefix")', source)
        self.assertIn('settings.get("outputThumbnailPrefix")', source)
        self.assertIn('full_key = f"{canonical_image_prefix}/{safe_name(job_id)}.png"', source)
        self.assertIn('thumb_key = f"{canonical_thumbnail_prefix}/{safe_name(job_id)}.jpg"', source)

    def test_old_single_prefix_contract_remains_only_as_rolling_deploy_fallback(self):
        source = inspect.getsource(media.finalize_image_outputs)
        self.assertIn('full_key = f"{prefix}/original/{safe_name(job_id)}.png"', source)
        self.assertIn('thumb_key = f"{prefix}/thumbnail/{safe_name(job_id)}.jpg"', source)
        self.assertIn('/scene_images', source)


if __name__ == "__main__":
    unittest.main()

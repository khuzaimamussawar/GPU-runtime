from pathlib import Path

from enhancer.src.gpu import image_batch_max_for_vram_mb, image_vram_class_gb


def test_storyboard_image_batch_contract_is_vram_bounded_and_dimension_safe():
    source = (Path(__file__).parents[1] / "src" / "fast_pipeline.py").read_text()
    assert "current_gpu_vram_mb()" in source
    assert "image_batch_max_for_vram_mb(vram_mb)" in source
    assert "len(images) > batch_max" in source
    assert "identical original width/height" in source
    assert "SOURCE_DIMENSIONS_MISMATCH" in source
    assert 'item.get("width")' in source
    assert 'item.get("height")' in source


def test_storyboard_image_batch_vram_tiers_match_control_plane_contract():
    assert image_batch_max_for_vram_mb(16 * 1024) == 5
    assert image_batch_max_for_vram_mb(20 * 1024) == 5
    assert image_batch_max_for_vram_mb(24 * 1024) == 10
    assert image_batch_max_for_vram_mb(32 * 1024) == 10
    assert image_batch_max_for_vram_mb(33 * 1024) == 20
    assert image_batch_max_for_vram_mb(48 * 1024) == 20
    assert image_batch_max_for_vram_mb(0) == 5
    assert image_vram_class_gb(24564) == 24

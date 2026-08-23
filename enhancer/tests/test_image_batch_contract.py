from pathlib import Path


def test_storyboard_image_batch_contract_is_bounded_and_dimension_safe():
    source = (Path(__file__).parents[1] / "src" / "fast_pipeline.py").read_text()
    assert "IMAGE_BATCH_MAX = 20" in source
    assert "len(images) > IMAGE_BATCH_MAX" in source
    assert "identical original width/height" in source
    assert "SOURCE_DIMENSIONS_MISMATCH" in source
    assert 'item.get("width")' in source
    assert 'item.get("height")' in source

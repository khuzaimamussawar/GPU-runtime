from enhancer.src.video_geometry import target_dimensions


def test_style_aspect_selects_exact_landscape_or_portrait_target_dimensions():
    assert target_dimensions("1080p", "16:9") == (1920, 1080)
    assert target_dimensions("1080p", "9:16") == (1080, 1920)
    assert target_dimensions("2k", "16:9") == (2560, 1440)
    assert target_dimensions("2k", "9:16") == (1440, 2560)
    assert target_dimensions("4k", "16:9") == (3840, 2160)
    assert target_dimensions("4k", "9:16") == (2160, 3840)

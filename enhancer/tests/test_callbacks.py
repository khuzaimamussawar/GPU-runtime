from enhancer.src.callbacks import EVENT_PATH, H3_EVENT_PATH, event_url


def test_enhancer_callbacks_use_h3_machine_ingress_from_origin():
    assert event_url("https://scene-builder.example.com") == (
        "https://scene-builder.example.com" + H3_EVENT_PATH
    )


def test_enhancer_callback_path_is_rewritten_to_h3_ingress():
    assert event_url("https://scene-builder.example.com" + EVENT_PATH) == (
        "https://scene-builder.example.com" + H3_EVENT_PATH
    )


def test_h3_callback_path_is_left_unchanged():
    target = "https://scene-builder.example.com" + H3_EVENT_PATH
    assert event_url(target) == target

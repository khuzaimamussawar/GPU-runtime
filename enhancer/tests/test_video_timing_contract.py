from pathlib import Path

from enhancer.src.video_parts import PreparedVideo


def _prepared(output_end_ms: float) -> PreparedVideo:
    return PreparedVideo(
        path=Path("selected.mkv"),
        parts=[],
        source_breaks_ms=[],
        raw_source_breaks_ms=[],
        timeline_parts=[{
            "sourceStartMs": 0.0,
            "sourceEndMs": output_end_ms,
            "outputStartMs": 0.0,
            "outputEndMs": output_end_ms,
            "timingBaked": 1.0,
        }],
        timing_baked=True,
    )


def test_director_target_frames_always_round_up():
    # 4.2 and 4.5 frames both need five actual CFR frames.  The output may
    # exceed the endpoint by less than one frame, but it must never be short.
    assert _prepared(87.5).minimum_output_frame_count(48) == 5
    assert _prepared(93.75).minimum_output_frame_count(48) == 5


def test_exact_director_frame_boundary_is_not_given_an_extra_frame():
    assert _prepared(100.0).minimum_output_frame_count(48) == 5


def test_linked_part_targets_use_the_cumulative_director_endpoint():
    prepared = _prepared(13_592)
    assert prepared.minimum_output_frame_count(48) == 653

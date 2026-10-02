"""Unit tests for src/detection/caption_module.py's pure caption-lookup
logic (get_caption_for_image) - no model loading or detection involved."""

from pathlib import Path

from src.detection.caption_module import get_caption_for_image

VIDEO_LABELS = {
    "shoplifting1": {
        "global_sequence_description": "global description for shoplifting1",
        "chronological_sequence_segments": [
            {"frame_range": "0-48", "scene_description": "setup phase caption"},
            {"frame_range": "49-96", "scene_description": "transition phase caption"},
            {"frame_range": "97-144", "scene_description": "resolution phase caption"},
        ],
    }
}


def test_get_caption_for_image_matches_the_right_segment():
    caption = get_caption_for_image(Path("shoplifting1_f0072.png"), VIDEO_LABELS)
    assert caption == "transition phase caption"


def test_get_caption_for_image_matches_first_segment_at_boundary():
    caption = get_caption_for_image(Path("shoplifting1_f0000.png"), VIDEO_LABELS)
    assert caption == "setup phase caption"


def test_get_caption_for_image_falls_back_to_global_description_outside_any_segment():
    caption = get_caption_for_image(Path("shoplifting1_f9999.png"), VIDEO_LABELS)
    assert caption == "global description for shoplifting1"


def test_get_caption_for_image_unknown_video_returns_placeholder():
    caption = get_caption_for_image(Path("unknown_video_f0010.png"), VIDEO_LABELS)
    assert "no VLM-labels record" in caption

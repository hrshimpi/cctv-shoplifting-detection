"""Unit tests for src/detection/prepare_yolo_data.py's video/label logic."""

from pathlib import Path

from src.detection.prepare_yolo_data import choose_val_videos, write_remapped_label


def test_choose_val_videos_holds_out_last_video_per_class():
    video_labels = {
        "not_shoplifting1": {"is_anomaly_sequence": False},
        "not_shoplifting2": {"is_anomaly_sequence": False},
        "shoplifting1": {"is_anomaly_sequence": True},
        "shoplifting2": {"is_anomaly_sequence": True},
    }

    val_videos = choose_val_videos(video_labels)

    assert val_videos == {"not_shoplifting2", "shoplifting2"}


def test_choose_val_videos_handles_a_single_class_present():
    video_labels = {"not_shoplifting1": {"is_anomaly_sequence": False}}
    assert choose_val_videos(video_labels) == {"not_shoplifting1"}


def test_write_remapped_label_strips_keypoints_and_remaps_class(tmp_path):
    src_label = tmp_path / "src.txt"
    keypoints = " ".join("0.1 0.2 2" for _ in range(17))
    src_label.write_text(f"0 0.5 0.5 0.2 0.3 {keypoints}\n0 0.1 0.1 0.05 0.05 {keypoints}\n", encoding="utf-8")
    dest_label = tmp_path / "dest.txt"

    write_remapped_label(src_label, dest_label, new_cls_id=1)

    lines = dest_label.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        parts = line.split()
        assert len(parts) == 5  # class + bbox only, no keypoints
        assert parts[0] == "1"  # remapped class id

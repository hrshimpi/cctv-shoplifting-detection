"""Unit tests for src/utils/dataset_discovery.py.

These exercise the discovery helpers against small synthetic directory
trees (via tmp_path) rather than the real dataset, so they run fast and
without needing data/ populated - exactly the "discover, don't hardcode"
logic this project leans on everywhere, tested in isolation.
"""

import json
from pathlib import Path

import yaml

from src.utils.dataset_discovery import (
    classify_image_by_video,
    find_annotation_files,
    find_file,
    find_images_dir,
    frame_number,
    labels_dir_for_images,
    list_images,
    load_class_config,
    load_vlm_video_labels,
    parse_yolo_label,
    resolve_split_dirs,
    video_stem_for_image,
)


def test_video_stem_for_image_strips_frame_suffix():
    assert video_stem_for_image(Path("shoplifting3_f0102.png")) == "shoplifting3"
    assert video_stem_for_image(Path("not_shoplifting1_f0000.png")) == "not_shoplifting1"


def test_frame_number_extracts_trailing_digits():
    assert frame_number(Path("shoplifting1_f0072.png")) == 72
    assert frame_number(Path("no_frame_suffix.png")) == -1


def test_labels_dir_for_images_mirrors_images_segment():
    images_dir = Path("data") / "CCTV_Shoplifting_Dataset" / "images"
    assert labels_dir_for_images(images_dir) == Path("data") / "CCTV_Shoplifting_Dataset" / "labels"


def test_labels_dir_for_images_falls_back_to_sibling():
    images_dir = Path("data") / "frames"
    assert labels_dir_for_images(images_dir) == Path("data") / "labels"


def test_classify_image_by_video_known_and_unknown():
    video_labels = {
        "shoplifting1": {"is_anomaly_sequence": True},
        "not_shoplifting1": {"is_anomaly_sequence": False},
    }
    assert classify_image_by_video(Path("shoplifting1_f0010.png"), video_labels) == "shoplifting"
    assert classify_image_by_video(Path("not_shoplifting1_f0010.png"), video_labels) == "not_shoplifting"
    assert classify_image_by_video(Path("unknown_video_f0010.png"), video_labels) is None


def test_find_file_locates_nested_file(tmp_path):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "data.yaml").write_text("names: [x]", encoding="utf-8")

    found = find_file(tmp_path, {"data.yaml"})
    assert found == nested / "data.yaml"
    assert find_file(tmp_path, {"classes.txt"}) is None


def test_find_images_dir_locates_dir_named_images(tmp_path):
    images_dir = tmp_path / "CCTV_Shoplifting_Dataset" / "images"
    images_dir.mkdir(parents=True)
    (tmp_path / "CCTV_Shoplifting_Dataset" / "annotated_images").mkdir()

    found = find_images_dir(tmp_path)
    assert found == images_dir


def test_find_annotation_files_matches_keyword_in_name(tmp_path):
    (tmp_path / "shoplifting1_vlm_meta.json").write_text("{}", encoding="utf-8")
    (tmp_path / "unrelated.json").write_text("{}", encoding="utf-8")

    found = find_annotation_files(tmp_path)
    assert found == [tmp_path / "shoplifting1_vlm_meta.json"]


def test_load_class_config_from_data_yaml(tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(yaml.dump({"names": {0: "not_shoplifting_person", 1: "shoplifting_person"}}), encoding="utf-8")

    names, cfg = load_class_config(data_yaml, None)
    assert names == ["not_shoplifting_person", "shoplifting_person"]
    assert cfg["names"]


def test_load_class_config_from_classes_txt(tmp_path):
    classes_txt = tmp_path / "classes.txt"
    classes_txt.write_text("not_shoplifting_person\nshoplifting_person\n", encoding="utf-8")

    names, cfg = load_class_config(None, classes_txt)
    assert names == ["not_shoplifting_person", "shoplifting_person"]
    assert cfg == {}


def test_load_class_config_returns_none_when_neither_exists():
    names, cfg = load_class_config(None, None)
    assert names is None
    assert cfg == {}


def test_load_vlm_video_labels_keys_by_video_stem(tmp_path):
    vlm_file = tmp_path / "shoplifting1_vlm_meta.json"
    vlm_file.write_text(json.dumps({"video_filename": "shoplifting1.mp4", "is_anomaly_sequence": True}), encoding="utf-8")

    video_labels = load_vlm_video_labels([vlm_file])
    assert video_labels == {"shoplifting1": {"video_filename": "shoplifting1.mp4", "is_anomaly_sequence": True}}


def test_parse_yolo_label_reads_bbox_and_ignores_trailing_keypoints(tmp_path):
    label_path = tmp_path / "frame.txt"
    # class + bbox, then 17 (x, y, visibility) pose keypoint triplets that
    # must be ignored - this is the real label shape this dataset ships.
    keypoints = " ".join("0.1 0.2 2" for _ in range(17))
    label_path.write_text(f"1 0.5 0.5 0.2 0.3 {keypoints}\n", encoding="utf-8")

    boxes = parse_yolo_label(label_path)
    assert boxes == [(1, 0.5, 0.5, 0.2, 0.3)]


def test_parse_yolo_label_missing_file_returns_empty_list(tmp_path):
    assert parse_yolo_label(tmp_path / "does_not_exist.txt") == []


def test_list_images_filters_by_extension(tmp_path):
    (tmp_path / "frame1.png").write_bytes(b"")
    (tmp_path / "frame2.jpg").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")

    images = list_images(tmp_path)
    assert {p.name for p in images} == {"frame1.png", "frame2.jpg"}


def test_resolve_split_dirs_falls_back_to_images_split_layout(tmp_path):
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "val").mkdir(parents=True)

    splits = resolve_split_dirs(tmp_path, None, {})
    assert splits == {
        "train": tmp_path / "images" / "train",
        "val": tmp_path / "images" / "val",
    }

"""
Loader for the RetailS pose dataset (TeCSAR-UNCC/RetailS).

This module discovers and parses whatever is actually present under a
RetailS root rather than assuming the course spec's summary matches
exactly - see README.md's "Dataset" section for the full writeup. Real
schema, confirmed on 2026-10-15 against the actual downloaded archive
(https://github.com/TeCSAR-UNCC/RetailS, Google Drive distribution):

  <subset>/pose/<train|test>/<camera_id>_<video_id>_<part>.json
    {
      "<person_id>": {
        "<frame_id>": {
          "keypoints": [x0, y0, c0, x1, y1, c1, ..., x16, y16, c16],
          "scores": null
        },
        ...
      },
      ...
    }
  51 floats per frame = 17 keypoints x (x, y, confidence), COCO-17 order
  per the dataset's paper. "scores" was present but null in every sample
  file checked.

  <subset>/gt/test_frame_mask/<camera_id>_<video_id>_<part>.npy
  uint8 array, one value per frame of the *video* (0=normal,
  1=anomalous) - indexed by frame, not by person. Only
  RetailS_test_realworld and RetailS_test_staged ship this; RetailS_train
  has no gt/ directory at all (confirmed: it's built for the paper's
  unsupervised/continual-adaptation setup, not a conventional labeled
  train split - 0 shoplifting events in train by design).

Real discrepancies from the course spec's summary, confirmed against the
actual files (not assumed):
  - No `bounding_box` field anywhere in the pose JSON.
  - No `camera_id` field inside the JSON - it's encoded in the filename
    (`<camera_id>_<video_id>_<part>.json`), not a per-record value.
  - The anomaly label is not an inline per-frame/per-person field; it's a
    separate per-video .npy array, frame-indexed only.
  - Per-person frame_ids are not contiguous (people enter/exit a track;
    confirmed e.g. person "2" in one train file only has frames
    1355-1420 while the file's person "1" has frames 0-208) - track
    discontinuity is real, not a hypothetical edge case to handle.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Train filenames have a trailing "_<part>" segment (long videos split
# into parts); test filenames (both realworld and staged) don't - this
# was NOT in the course spec summary and only surfaced by actually
# running the loader against the real files (confirmed 2026-10-15: train
# e.g. "1_041812_1.json", test e.g. "6_1050002.json"). `part` is
# therefore optional and defaults to "1" when absent, not assumed present.
FILENAME_RE = re.compile(r"^(?P<camera_id>\d+)_(?P<video_id>\d+)(?:_(?P<part>\d+))?$")

# Subset directory names as actually found in the downloaded archive.
SUBSET_DIRS = {
    "train": "RetailS_train",
    "test_realworld": "RetailS_test_realworld",
    "test_staged": "RetailS_test_staged",
}


@dataclass
class PoseFrame:
    frame_id: int
    keypoints: np.ndarray  # shape (17, 3): x, y, confidence
    scores: float | None


@dataclass
class PersonTrack:
    person_id: str
    camera_id: str
    video_id: str
    part: str
    source_file: Path
    frames: dict[int, PoseFrame]  # frame_id -> PoseFrame; not necessarily contiguous

    @property
    def frame_ids_sorted(self) -> list[int]:
        return sorted(self.frames)


def discover_retails_root(data_dir: Path) -> Path | None:
    """Find the directory directly containing RetailS_train/ etc.,
    wherever it landed under data_dir (handles the extra 'RetailS/'
    nesting level the official archive ships with)."""
    for candidate in data_dir.rglob(SUBSET_DIRS["train"]):
        if candidate.is_dir():
            return candidate.parent
    return None


def parse_filename(json_path: Path) -> tuple[str, str, str]:
    """Return (camera_id, video_id, part) parsed from a RetailS pose
    filename. Raises ValueError rather than silently guessing if a file
    doesn't match the discovered naming convention."""
    m = FILENAME_RE.match(json_path.stem)
    if not m:
        raise ValueError(
            f"{json_path.name} doesn't match the expected "
            "'<camera_id>_<video_id>_<part>.json' convention"
        )
    return m["camera_id"], m["video_id"], m["part"] or "1"


def find_pose_files(subset_dir: Path) -> list[Path]:
    """Find all pose JSON files under one RetailS subset directory
    (e.g. RetailS_train), regardless of the train/test split folder
    name inside it."""
    return sorted((subset_dir / "pose").rglob("*.json"))


def gt_dir_for_subset(subset_dir: Path) -> Path | None:
    gt_dir = subset_dir / "gt" / "test_frame_mask"
    return gt_dir if gt_dir.is_dir() else None


def label_path_for_pose_file(pose_path: Path, gt_dir: Path | None) -> Path | None:
    if gt_dir is None:
        return None
    candidate = gt_dir / f"{pose_path.stem}.npy"
    return candidate if candidate.exists() else None


def load_pose_file(json_path: Path) -> list[PersonTrack]:
    """Load one RetailS pose JSON into a PersonTrack per person_id it contains."""
    camera_id, video_id, part = parse_filename(json_path)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    tracks = []
    for person_id, frame_dict in data.items():
        frames: dict[int, PoseFrame] = {}
        for frame_id_str, record in frame_dict.items():
            kp_flat = record.get("keypoints")
            if not kp_flat or len(kp_flat) % 3 != 0:
                continue
            keypoints = np.asarray(kp_flat, dtype=np.float32).reshape(-1, 3)
            frames[int(frame_id_str)] = PoseFrame(
                frame_id=int(frame_id_str),
                keypoints=keypoints,
                scores=record.get("scores"),
            )
        tracks.append(
            PersonTrack(
                person_id=person_id,
                camera_id=camera_id,
                video_id=video_id,
                part=part,
                source_file=json_path,
                frames=frames,
            )
        )
    return tracks


def load_anomaly_labels(npy_path: Path | None) -> np.ndarray | None:
    if npy_path is None or not npy_path.exists():
        return None
    return np.load(npy_path)

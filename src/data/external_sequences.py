"""Auditable external-video windows; video class names are not temporal labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.data.preprocessing import build_windows, preprocess_track
from src.data.retails import PersonTrack, PoseFrame
from src.features.motion_features import build_feature_vector, feature_dim

SCHEMA_VERSION = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_hashes():
    src = Path(__file__).resolve().parents[1]
    return {name: sha256(src / name) for name in
            ["data/preprocessing.py", "data/external_sequences.py", "features/motion_features.py"]}


def build_sequence_arrays(records, source: Path, cfg, min_visible_fraction=0.5, min_anchor_fraction=0.5):
    """Return aligned windows/features, preserving masks and source-frame IDs.

    Quality thresholds are technical filters only. They cannot establish
    correct person detection, identity continuity or a theft event.
    """
    for value in [min_visible_fraction, min_anchor_fraction]:
        if not 0 <= value <= 1:
            raise ValueError("Quality fractions must be in [0, 1]")
    length, stride = cfg["window_length"], cfg["stride"]
    if length < 1 or stride < 1:
        raise ValueError("Window length and stride must be positive")
    kept, track_ids, raw_visibility, usable_visibility, anchors = [], [], [], [], []
    track_rows = []
    for tid in sorted(records, key=str):
        track = PersonTrack(str(tid), "external_demo", source.stem, "1", source,
                            {int(fid): PoseFrame(int(fid), np.asarray(record["keypoints"], dtype=np.float32).reshape(17, 3), None)
                             for fid, record in records[tid].items()})
        processed = preprocess_track(track, confidence_threshold=cfg["confidence_threshold"],
                                     max_interpolation_gap=cfg["max_interpolation_gap"],
                                     smoothing_window=cfg["smoothing_window"])
        windows = build_windows(processed["frame_ids"], processed["keypoints"], processed["visibility_mask"], length, stride)
        accepted = 0
        for window in windows:
            raw_fraction = float((window["keypoints"][..., 2] >= cfg["confidence_threshold"]).mean())
            visible_fraction = float(window["visibility_mask"].mean())
            anchor_fraction = float(window["visibility_mask"][:, [5, 6, 11, 12]].all(axis=1).mean())
            if visible_fraction < min_visible_fraction or anchor_fraction < min_anchor_fraction:
                continue
            kept.append(window)
            track_ids.append(str(tid))
            raw_visibility.append(raw_fraction)
            usable_visibility.append(visible_fraction)
            anchors.append(anchor_fraction)
            accepted += 1
        track_rows.append({"track_id": str(tid), "observations": len(track.frames),
                           "candidate_windows": len(windows), "retained_windows": accepted,
                           "rejected_windows": len(windows) - accepted})
    points = np.stack([w["keypoints"] for w in kept]) if kept else np.empty((0, length, 17, 3), dtype=np.float32)
    masks = np.stack([w["visibility_mask"] for w in kept]) if kept else np.empty((0, length, 17), dtype=bool)
    frame_ids = np.array([np.arange(w["start_frame"], w["end_frame"] + 1) for w in kept], dtype=np.int64).reshape(-1, length)
    arrays = {
        "keypoints": points, "visibility_mask": masks, "frame_ids": frame_ids,
        "track_ids": np.asarray(track_ids, dtype=str),
        "raw_visible_fraction": np.asarray(raw_visibility),
        "usable_visible_fraction": np.asarray(usable_visibility),
        "anchor_fraction": np.asarray(anchors),
        "features_a": np.stack([build_feature_vector(w[:, :, :2]) for w in points]) if kept else np.empty((0, length, feature_dim())),
        "features_b": np.stack([build_feature_vector(w[:, :, :2], use_velocity=True) for w in points]) if kept else np.empty((0, length, feature_dim(True))),
    }
    return arrays, track_rows


def validate_sequence_arrays(arrays, source_frames: int, cfg, min_visible_fraction=0.5, min_anchor_fraction=0.5):
    points = arrays["keypoints"]
    n, length = len(points), cfg["window_length"]
    expected_shapes = {"keypoints": (n, length, 17, 3), "visibility_mask": (n, length, 17),
                       "frame_ids": (n, length), "track_ids": (n,),
                       "raw_visible_fraction": (n,), "usable_visible_fraction": (n,), "anchor_fraction": (n,),
                       "features_a": (n, length, feature_dim()), "features_b": (n, length, feature_dim(True))}
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"Wrong {name} shape: {arrays[name].shape}; expected {shape}")
        if name != "track_ids" and not np.isfinite(arrays[name]).all():
            raise ValueError(f"Non-finite {name}")
    if arrays["visibility_mask"].dtype != np.bool_ or not np.issubdtype(arrays["frame_ids"].dtype, np.integer):
        raise ValueError("Masks must be boolean and frame IDs integer")
    ids = arrays["frame_ids"]
    if n and ((ids < 0).any() or (ids >= source_frames).any() or not (np.diff(ids, axis=1) == 1).all()):
        raise ValueError("Window crosses a frame gap or source boundary")
    if any(not str(tid) for tid in arrays["track_ids"]):
        raise ValueError("Missing track ID")
    if ((points[..., 2] < 0) | (points[..., 2] > 1)).any():
        raise ValueError("Invalid confidence")
    visible = arrays["visibility_mask"].mean(axis=(1, 2)) if n else np.empty(0)
    anchor = arrays["visibility_mask"][:, :, [5, 6, 11, 12]].all(axis=2).mean(axis=1) if n else np.empty(0)
    raw = (points[..., 2] >= cfg["confidence_threshold"]).mean(axis=(1, 2)) if n else np.empty(0)
    for name, actual in [("usable_visible_fraction", visible), ("anchor_fraction", anchor), ("raw_visible_fraction", raw)]:
        if not np.allclose(arrays[name], actual):
            raise ValueError(f"Incorrect {name}")
    if (visible < min_visible_fraction).any() or (anchor < min_anchor_fraction).any():
        raise ValueError("Window fails configured visibility/anchor quality threshold")
    for i, window in enumerate(points):
        for name, velocity in [("features_a", False), ("features_b", True)]:
            expected = build_feature_vector(window[:, :, :2], use_velocity=velocity)
            if not np.allclose(arrays[name][i], expected):
                raise ValueError(f"{name} is not aligned with keypoints")
    return n


def export_sequences(pose_dir: Path, out_dir: Path, video_label: str, cfg,
                     min_visible_fraction=0.5, min_anchor_fraction=0.5):
    summary = json.loads((pose_dir / "summary.json").read_text(encoding="utf-8"))
    source = Path(summary["source_video"])
    provenance = {"schema_version": SCHEMA_VERSION, "source_sha256": sha256(source),
                  "poses_sha256": sha256(pose_dir / "poses.json"), "config": cfg,
                  "implementation_sha256": implementation_hashes(),
                  "video_label": video_label, "min_visible_fraction": min_visible_fraction,
                  "min_anchor_fraction": min_anchor_fraction}
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    marker = out_dir / "manifest.json"
    if marker.exists():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing["fingerprint"] != fingerprint:
            raise ValueError(f"Stale export at {out_dir}; select a new --out-root.")
        return verify_sequences(out_dir)
    records = json.loads((pose_dir / "poses.json").read_text(encoding="utf-8"))
    arrays, track_rows = build_sequence_arrays(records, source, cfg, min_visible_fraction, min_anchor_fraction)
    validate_sequence_arrays(arrays, summary["source_frames"], cfg, min_visible_fraction, min_anchor_fraction)
    out_dir.mkdir(parents=True, exist_ok=True)
    temp = out_dir / "windows.tmp"
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(out_dir / "windows.npz")
    candidate_count = sum(t["candidate_windows"] for t in track_rows)
    manifest = {**provenance, "fingerprint": fingerprint, "source_video": str(source.resolve()),
                "pose_dir": str(pose_dir.resolve()), "source_frames": summary["source_frames"],
                "fps": summary["fps"], "detection_confidence": summary["confidence_threshold"],
                "track_frame_coverage": summary.get("track_frame_coverage"),
                "candidate_windows": candidate_count, "retained_windows": len(arrays["keypoints"]),
                "rejected_windows": candidate_count - len(arrays["keypoints"]),
                "tracks": track_rows, "windows_sha256": sha256(out_dir / "windows.npz"),
                "manual_review_status": "pending", "label_scope": "video only; no per-window anomaly ground truth",
                "purpose": "external development/demo; not a training/test split",
                "preprocessing": "smoothing isolated to contiguous frame-ID segments"}
    temp = out_dir / "manifest.tmp"
    temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temp.replace(marker)
    return verify_sequences(out_dir)


def verify_sequences(out_dir: Path):
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    provenance = {key: manifest[key] for key in ["schema_version", "source_sha256", "poses_sha256", "config", "implementation_sha256",
                  "video_label", "min_visible_fraction", "min_anchor_fraction"]}
    expected_fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    if manifest["schema_version"] != SCHEMA_VERSION or expected_fingerprint != manifest["fingerprint"]:
        raise ValueError("Manifest configuration/provenance fingerprint mismatch")
    if manifest["implementation_sha256"] != implementation_hashes():
        raise ValueError("Preprocessing implementation changed; rebuild in a new output directory")
    if sha256(Path(manifest["source_video"])) != manifest["source_sha256"]:
        raise ValueError("Source video changed since export")
    pose_file = Path(manifest["pose_dir"]) / "poses.json"
    if sha256(pose_file) != manifest["poses_sha256"]:
        raise ValueError("Pose source changed since export")
    npz = out_dir / "windows.npz"
    if sha256(npz) != manifest["windows_sha256"]:
        raise ValueError("Window file checksum mismatch")
    with np.load(npz, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    count = validate_sequence_arrays(arrays, manifest["source_frames"], manifest["config"],
                                     manifest["min_visible_fraction"], manifest["min_anchor_fraction"])
    if count != manifest["retained_windows"]:
        raise ValueError("Retained-window count mismatch")
    records = json.loads(pose_file.read_text(encoding="utf-8"))
    seen = set()
    for tid, ids in zip(arrays["track_ids"], arrays["frame_ids"]):
        if tid not in records or any(str(fid) not in records[tid] for fid in ids):
            raise ValueError("Window frame/track provenance missing from pose source")
        key = (tid, int(ids[0]))
        if key in seen:
            raise ValueError("Duplicate window")
        seen.add(key)
    return manifest

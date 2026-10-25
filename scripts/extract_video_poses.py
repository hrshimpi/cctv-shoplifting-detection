"""Extract tracked COCO-17 poses and a skeleton preview without an anomaly model.

Example:
    python scripts/extract_video_poses.py --video data/external_video/videos/shoplifting/Shoplifting033_x264.mp4
    python scripts/extract_video_poses.py --verify-only --out-dir outputs/poses/Shoplifting033_x264
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import yaml

from src.data.preprocessing import build_windows, preprocess_track
from src.demo.render import draw_label, draw_skeleton
from src.demo.video_pose import DEFAULT_YOLO_CHECKPOINT, detect_and_track

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def validate_poses(poses, frame_count):
    """Reject malformed coordinates and frame IDs before accepting an export."""
    if not poses:
        raise ValueError("No person tracks detected; inspect video/threshold before proceeding.")
    tracked_frames = set()
    observations = 0
    for frames in poses.values():
        if not frames:
            raise ValueError("Empty person track")
        for frame_id, record in frames.items():
            if not 0 <= int(frame_id) < frame_count:
                raise ValueError(f"Frame ID outside source video: {frame_id}")
            points = np.asarray(record["keypoints"], dtype=float)
            if points.shape != (51,) or not np.isfinite(points).all():
                raise ValueError("Expected 17 finite (x, y, confidence) keypoints")
            if ((points[2::3] < 0) | (points[2::3] > 1)).any():
                raise ValueError("Keypoint confidence outside [0, 1]")
            tracked_frames.add(int(frame_id))
            observations += 1
    return {"tracks": len(poses), "person_frame_observations": observations,
            "frames_with_tracks": len(tracked_frames)}


def verify_export(out_dir):
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    poses = json.loads((out_dir / "poses.json").read_text(encoding="utf-8"))
    counts = validate_poses(poses, summary["source_frames"])
    for name, value in counts.items():
        if summary[name] != value:
            raise ValueError(f"Summary mismatch for {name}")
    cap = cv2.VideoCapture(str(out_dir / "skeletons.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = summary["frame_size"]
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (height, width):
            raise ValueError("Preview dimensions differ from source")
        decoded += 1
    cap.release()
    if decoded != summary["source_frames"] or not np.isclose(fps, summary["fps"], atol=0.01):
        raise ValueError("Preview frame count/FPS differs from source")
    if cv2.imread(str(out_dir / "preview.jpg")) is None:
        raise ValueError("Preview image cannot be decoded")
    print(f"PASS: {counts['tracks']} tracks; {decoded} preview frames at {fps:g} FPS; finite COCO-17 keypoints.")
    return summary


def extract(video, out_dir, checkpoint, conf_threshold, img_size=640):
    if not video.is_file():
        raise FileNotFoundError(video)
    if (out_dir / "summary.json").exists():
        raise FileExistsError(f"Export already exists at {out_dir}; use --verify-only or choose another --out-dir.")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(f"Extracting poses from {video.name}...", flush=True)
    tracks, fps, size = detect_and_track(video, checkpoint, conf_threshold, img_size=img_size)
    cap = cv2.VideoCapture(str(video))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    poses = {str(tid): {str(fid): {"keypoints": track.frames[fid].keypoints.reshape(-1).tolist(), "scores": None}
                       for fid in track.frame_ids_sorted} for tid, track in tracks.items()}
    counts = validate_poses(poses, frame_count)
    cfg = yaml.safe_load((PROJECT_ROOT / "configs/preprocessing.yaml").read_text(encoding="utf-8"))
    window_count = 0
    for track in tracks.values():
        processed = preprocess_track(track, confidence_threshold=cfg["confidence_threshold"],
                                     max_interpolation_gap=cfg["max_interpolation_gap"],
                                     smoothing_window=cfg["smoothing_window"])
        if not np.isfinite(processed["keypoints"]).all():
            raise ValueError("Preprocessing produced non-finite values")
        windows = build_windows(processed["frame_ids"], processed["keypoints"], processed["visibility_mask"],
                                window_length=cfg["window_length"], stride=cfg["stride"])
        window_count += len(windows)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(str(out_dir / "skeletons.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create skeleton preview video")
    frames_with_tracks = sorted({fid for track in tracks.values() for fid in track.frames})
    preview_frame = frames_with_tracks[len(frames_with_tracks) // 2]
    decoded = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for tid, track in tracks.items():
                if decoded in track.frames:
                    keypoints = track.frames[decoded].keypoints
                    draw_skeleton(frame, keypoints, (255, 200, 0), conf_threshold)
                    draw_label(frame, keypoints, f"ID {tid}", (255, 200, 0))
            writer.write(frame)
            if decoded == preview_frame and not cv2.imwrite(str(out_dir / "preview.jpg"), frame):
                raise RuntimeError("Cannot write preview image")
            decoded += 1
    finally:
        cap.release()
        writer.release()
    if decoded != frame_count:
        raise ValueError(f"Decoded {decoded} frames but source reports {frame_count}")
    summary = {"source_video": str(video.resolve()), "checkpoint": str(checkpoint.resolve()),
               "fps": fps, "frame_size": list(size), "source_frames": frame_count,
               **counts, "track_frame_coverage": counts["frames_with_tracks"] / frame_count,
               "preprocessing_windows": window_count, "preprocessing_config": cfg,
               "confidence_threshold": conf_threshold,
               "inference_img_size": img_size,
               "note": "Pose/tracking preview only; no anomaly classification or image-level theft labels."}
    (out_dir / "poses.json").write_text(json.dumps(poses, allow_nan=False), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    verify_export(out_dir)
    print(f"Preprocessing smoke test: {window_count} temporal windows. Outputs: {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_YOLO_CHECKPOINT)
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    parser.add_argument("--img-size", type=int, default=640, help="YOLO inference resolution; saved coordinates remain source pixels")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.conf_threshold <= 1:
        parser.error("--conf-threshold must be in (0, 1]")
    if args.img_size < 32 or args.img_size % 32:
        parser.error("--img-size must be a positive multiple of 32")
    if args.verify_only:
        if args.out_dir is None:
            parser.error("--verify-only requires --out-dir")
        verify_export(args.out_dir)
    else:
        if args.video is None:
            parser.error("--video is required")
        extract(args.video, args.out_dir or PROJECT_ROOT / "outputs/poses" / args.video.stem,
                args.checkpoint, args.conf_threshold, args.img_size)


if __name__ == "__main__":
    main()

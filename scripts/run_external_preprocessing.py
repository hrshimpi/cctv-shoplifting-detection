"""Batch external-video pose extraction and quality-filtered temporal exports.

    python scripts/run_external_preprocessing.py --extract-missing
    python scripts/run_external_preprocessing.py --verify-only
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "outputs/yolo_settings"))

import yaml

from scripts.extract_video_poses import extract, verify_export
from src.data.external_sequences import export_sequences, verify_sequences
from src.demo.video_pose import DEFAULT_YOLO_CHECKPOINT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, default=ROOT / "data/external_video/videos")
    parser.add_argument("--pose-root", type=Path, default=ROOT / "outputs/poses_batch/conf05")
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/preprocessing.yaml")
    parser.add_argument("--profiles", type=Path, help="Optional per-video model/threshold/pose-folder overrides")
    parser.add_argument("--detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-visible-fraction", type=float, default=0.5)
    parser.add_argument("--min-anchor-fraction", type=float, default=0.5)
    parser.add_argument("--extract-missing", action="store_true", help="Run YOLO only for missing pose exports")
    parser.add_argument("--verify-only", action="store_true", help="Check existing exports without running inference")
    args = parser.parse_args()
    if args.profiles and args.out_root is None:
        parser.error("--profiles requires an explicit --out-root to preserve the baseline")
    if args.out_root is None:
        args.out_root = ROOT / "outputs/external_sequences/conf05"
    if args.verify_only and args.extract_missing:
        parser.error("--verify-only cannot be combined with --extract-missing")
    if not 0 < args.detection_confidence <= 1 or not all(0 <= v <= 1 for v in [args.min_visible_fraction, args.min_anchor_fraction]):
        parser.error("Detection confidence must be in (0,1]; quality fractions in [0,1]")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    profiles = yaml.safe_load(args.profiles.read_text(encoding="utf-8")).get("videos", {}) if args.profiles else {}
    videos = []
    for label in ["normal", "shoplifting"]:
        found = sorted((args.video_root / label).glob("*.mp4"))
        if not found:
            raise SystemExit(f"No {label} videos under {args.video_root}")
        videos.extend((label, p) for p in found)
    if set(profiles) - {video.name for _, video in videos}:
        raise ValueError("Profile names do not match the current video inventory")
    rows = []
    for index, (label, video) in enumerate(videos, 1):
        profile = profiles.get(video.name, {})
        if set(profile) - {"pose_dir", "checkpoint", "detection_confidence", "img_size"}:
            raise ValueError(f"Unknown settings in profile for {video.name}")
        pose_dir = ROOT / profile["pose_dir"] if "pose_dir" in profile else args.pose_root / label / video.stem
        confidence = float(profile.get("detection_confidence", args.detection_confidence))
        checkpoint = ROOT / profile["checkpoint"] if "checkpoint" in profile else DEFAULT_YOLO_CHECKPOINT
        img_size = int(profile.get("img_size", 640))
        if not 0 < confidence <= 1 or img_size < 32 or img_size % 32:
            raise ValueError(f"Invalid inference settings in profile for {video.name}")
        out_dir = args.out_root / label / video.stem
        print(f"[{index}/{len(videos)}] {label}/{video.name}", flush=True)
        if not args.verify_only:
            if not (pose_dir / "summary.json").exists():
                if not args.extract_missing:
                    raise SystemExit(f"Missing pose export: {pose_dir}. Use --extract-missing to generate it.")
                extract(video, pose_dir, checkpoint, confidence, img_size)
            summary = verify_export(pose_dir)
        summary = json.loads((pose_dir / "summary.json").read_text(encoding="utf-8"))
        if (Path(summary["source_video"]).resolve() != video.resolve()
                or summary["confidence_threshold"] != confidence
                or Path(summary["checkpoint"]).resolve() != checkpoint.resolve()
                or summary.get("inference_img_size", 640) != img_size):
            raise ValueError(f"Pose export source/model/settings do not match: {pose_dir}")
        manifest = verify_sequences(out_dir) if args.verify_only else export_sequences(
            pose_dir, out_dir, label, cfg, args.min_visible_fraction, args.min_anchor_fraction)
        if Path(manifest["source_video"]).resolve() != video.resolve() or manifest["video_label"] != label:
            raise ValueError("Sequence export is associated with the wrong source video/class")
        if manifest["config"] != cfg or manifest["detection_confidence"] != confidence:
            raise ValueError("Sequence export uses different preprocessing/detection settings")
        if manifest["min_visible_fraction"] != args.min_visible_fraction or manifest["min_anchor_fraction"] != args.min_anchor_fraction:
            raise ValueError("Sequence export uses different quality settings")
        rows.append({"video": video.name, "video_label": label, "source_frames": manifest["source_frames"],
                     "tracks": len(manifest["tracks"]), "candidate_windows": manifest["candidate_windows"],
                     "retained_windows": manifest["retained_windows"], "rejected_windows": manifest["rejected_windows"],
                     "track_frame_coverage": manifest.get("track_frame_coverage"),
                     "checkpoint": str(checkpoint.resolve()), "detection_confidence": confidence, "img_size": img_size,
                     "export_dir": str(out_dir.resolve()), "manual_review_status": manifest["manual_review_status"]})
        print(f"PASS: {manifest['retained_windows']}/{manifest['candidate_windows']} windows retained; manual review pending.", flush=True)
    report = {"technical_checks_passed": True, "video_count": len(rows),
              "retained_windows": sum(r["retained_windows"] for r in rows), "videos": rows,
              "config": cfg, "detection_confidence": args.detection_confidence,
              "profiles": profiles,
              "min_visible_fraction": args.min_visible_fraction, "min_anchor_fraction": args.min_anchor_fraction,
              "training_ready": False, "note": "External video data with video-level labels only; manual scene/track review is pending."}
    args.out_root.mkdir(parents=True, exist_ok=True)
    temp = args.out_root / "batch_report.tmp"
    temp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temp.replace(args.out_root / "batch_report.json")
    print(f"PASS: {len(rows)} videos; {report['retained_windows']} retained windows. Report: {args.out_root / 'batch_report.json'}")


if __name__ == "__main__":
    main()

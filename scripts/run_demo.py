#!/usr/bin/env python
"""
Raw-video shoplifting-anomaly demo: person detection/tracking + 2D pose
(Ultralytics YOLO Pose, pretrained, with its built-in ByteTrack) ->
this project's own preprocessing -> a frozen, pretrained MotionBERT
2D->3D lift -> 3D motion/geometry features -> the best-performing
trained BiGRU autoencoder from step 4 (Experiment D by default) -> an
anomaly score per completed window, rendered back onto the video as an
annotated output. Standalone, runs locally - no notebook required.
Nothing here is trained from scratch: YOLO Pose and MotionBERT are both
used pretrained/frozen, and the BiGRU is the checkpoint
scripts/train.py already produced.

Usage:
    python scripts/run_demo.py --video data/external_video/videos/shoplifting/some_clip.mp4
    python scripts/run_demo.py --video <path> --experiment e --out outputs/demo/annotated.mp4

The input video is not included in this repo - see README.md "Dataset"
for how to obtain UCF-Crime clips and where to place them. This path is
for external validation/demo purposes only, not a benchmark - see
README.md "Limitations".
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.preprocessing import build_windows, preprocess_track  # noqa: E402
from src.demo.render import draw_label, draw_skeleton, status_color, status_text  # noqa: E402
from src.demo.video_pose import DEFAULT_YOLO_CHECKPOINT, detect_and_track  # noqa: E402
from src.evaluation.metrics import choose_threshold  # noqa: E402
from src.features.motion_features import build_feature_vector_3d, feature_dim_3d  # noqa: E402
from src.models.bigru_attention import BiGRUAttentionAutoencoder  # noqa: E402
from src.models.bigru_autoencoder import BiGRUAutoencoder  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MOTIONBERT_CHECKPOINT = (
    PROJECT_ROOT / "third_party" / "MotionBERT" / "checkpoint" / "pose3d" / "MB_ft_h36m" / "best_epoch.bin"
)
EXPERIMENT_CONFIGS = {
    "c": PROJECT_ROOT / "configs" / "experiment_c.yaml",
    "d": PROJECT_ROOT / "configs" / "experiment_d.yaml",
    "e": PROJECT_ROOT / "configs" / "experiment_e.yaml",
}
MODEL_CLASSES = {"bigru": BiGRUAutoencoder, "bigru_attention": BiGRUAttentionAutoencoder}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--out", default=None, help="Output annotated video path (default: outputs/demo/<video_stem>_annotated.mp4)")
    parser.add_argument(
        "--experiment", default="d", choices=sorted(EXPERIMENT_CONFIGS),
        help="Trained 3D experiment to score with - D is the best-performing one from step 4 (see README.md Results)",
    )
    parser.add_argument("--yolo-checkpoint", default=str(DEFAULT_YOLO_CHECKPOINT))
    parser.add_argument("--motionbert-checkpoint", default=str(DEFAULT_MOTIONBERT_CHECKPOINT))
    parser.add_argument("--conf-threshold", type=float, default=0.3, help="YOLO detection confidence threshold")
    return parser.parse_args()


def load_experiment(experiment: str):
    out_dir = OUTPUTS_DIR / "checkpoints" / f"experiment_{experiment}"
    checkpoint = torch.load(out_dir / "model.pt", weights_only=False)
    cfg = checkpoint["config"]
    model_cls = MODEL_CLASSES[cfg.get("model", "bigru")]
    input_dim = feature_dim_3d(cfg["use_velocity"], cfg["use_acceleration"], cfg["use_geometry"], cfg["use_angles"])
    model = model_cls(input_dim=input_dim, hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    train_errors = np.load(out_dir / "train_reconstruction_errors.npy")
    threshold = choose_threshold(train_errors, percentile=cfg["threshold_percentile"])
    return model, cfg, threshold


def score_tracks(
    tracks: dict, cfg: dict, model_3d, model, threshold: float
) -> dict[int, list[tuple[int, int, float, bool]]]:
    """For every track, build windows exactly as src/experiments/dataset.py
    does for training/evaluation (apply_normalization=False - MotionBERT
    wants its own normalization, see src/motionbert/lift.py), lift each
    to 3D, build the experiment's feature vector, and score it with the
    trained autoencoder.

    Returns {track_id: [(start_frame, end_frame, score, is_anomalous), ...]}.
    """
    from src.motionbert.lift import lift_2d_to_3d

    results: dict[int, list[tuple[int, int, float, bool]]] = {}
    for track_id, track in tracks.items():
        processed = preprocess_track(
            track,
            confidence_threshold=cfg["confidence_threshold"],
            max_interpolation_gap=cfg["max_interpolation_gap"],
            smoothing_window=cfg["smoothing_window"],
            apply_normalization=False,
        )
        windows = build_windows(
            processed["frame_ids"], processed["keypoints"], processed["visibility_mask"],
            window_length=cfg["window_length"], stride=cfg["stride"],
        )
        track_results = []
        for w in windows:
            lifted = lift_2d_to_3d(model_3d, w["keypoints"])
            feature_vector = build_feature_vector_3d(
                lifted, cfg["use_velocity"], cfg["use_acceleration"], cfg["use_geometry"], cfg["use_angles"]
            )
            x = torch.from_numpy(feature_vector).float().unsqueeze(0)
            with torch.no_grad():
                score = model.reconstruction_error(x).item()
            track_results.append((w["start_frame"], w["end_frame"], score, score >= threshold))
        results[track_id] = track_results
    return results


def render_output(
    video_path: Path, out_path: Path, tracks: dict, track_results: dict,
    fps: float, frame_size: tuple[int, int], conf_threshold: float,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for track_id, track in tracks.items():
            if frame_idx not in track.frames:
                continue
            kpts = track.frames[frame_idx].keypoints
            current = None
            for start, end, score, is_anom in track_results.get(track_id, []):
                if end <= frame_idx:
                    current = (score, is_anom)
            color = status_color(current[1] if current else None)
            draw_skeleton(frame, kpts, color, conf_threshold)
            draw_label(frame, kpts, status_text(str(track_id), current[0] if current else None, current[1] if current else None), color)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(
            f"No video at {video_path}. See README.md 'Dataset' for how to obtain UCF-Crime clips and where to place them."
        )

    out_path = Path(args.out) if args.out else OUTPUTS_DIR / "demo" / f"{video_path.stem}_annotated.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running YOLO Pose + ByteTrack on {video_path}...")
    t0 = time.time()
    tracks, fps, frame_size = detect_and_track(video_path, checkpoint_path=args.yolo_checkpoint, conf_threshold=args.conf_threshold)
    print(f"Detected {len(tracks)} track(s) in {time.time() - t0:.1f}s")
    if not tracks:
        print("No persons detected/tracked in this video - nothing to score or render.")
        return

    print("Loading MotionBERT and the trained anomaly detector...")
    from src.motionbert.lift import load_motionbert

    model_3d = load_motionbert(args.motionbert_checkpoint)
    model, cfg, threshold = load_experiment(args.experiment)
    print(f"Using experiment_{args.experiment} (threshold={threshold:.6f})")

    print("Lifting to 3D and scoring each track's windows...")
    t0 = time.time()
    track_results = score_tracks(tracks, cfg, model_3d, model, threshold)
    for track_id, windows in track_results.items():
        print(f"  track {track_id}: {len(windows)} window(s), {sum(1 for *_, a in windows if a)} flagged anomalous")
    print(f"Scored in {time.time() - t0:.1f}s")

    print(f"Rendering annotated video to {out_path}...")
    render_output(video_path, out_path, tracks, track_results, fps, frame_size, args.conf_threshold)
    print("Done.")


if __name__ == "__main__":
    main()

"""
Demo/validation run of the classical SSIM/ROI motion baseline against
real clips from the downloaded dataset.

Picks a fixed ROI over the shelf/table area for two real camera setups
(hand-picked by looking at actual frames from the dataset - see the
ROI_CONFIGS below and the README's baseline section), runs run_on_video()
on real clips, and reports where alerts start/stop relative to the end of
the clip. This is evidence the mechanism works on real footage, not a
rigorous benchmark.

Run after `python src/detection/prepare_yolo_dataset.py` (or just
download_dataset.py - this only needs the raw videos/ folder).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.classical_cv.motion_baseline import run_on_video  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "CCTV_Shoplifting_Dataset" / "videos"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classical_baseline"

# Hand-picked by looking at the actual frames (README has the before/after
# stills). Picking a shelf/display-case ROI is inherently a per-camera,
# human judgment call - that's exactly what select_roi_interactive.py is
# for on a real deployment; these are its manual equivalent for this demo.
ROI_CONFIGS = {
    "electronics_table": (175, 343, 75, 65),  # not_shoplifting1 / shoplifting1
    "hardware_aisle": (325, 255, 90, 90),  # not_shoplifting2 / shoplifting2
}

CLIPS = [
    ("not_shoplifting1", "electronics_table", "fixed"),
    ("shoplifting1", "electronics_table", "fixed"),
    ("shoplifting1", "electronics_table", "rolling"),
    ("shoplifting2", "hardware_aisle", "fixed"),
]


def main() -> None:
    header = (
        f"{'video':<18}{'mode':<9}{'frames':>7}{'alerts':>8}{'avg_sim':>9}"
        f"   first_alert  last_alert  recovered_before_end?"
    )
    print(header)
    print("-" * len(header))

    for video_stem, roi_key, mode in CLIPS:
        video_path = VIDEOS_DIR / f"{video_stem}.mp4"
        roi = ROI_CONFIGS[roi_key]
        run_dir = OUTPUT_DIR / f"{video_stem}_{mode}"

        detector = run_on_video(
            video_path,
            roi=roi,
            threshold=0.8,
            reference_mode=mode,
            output_video_path=run_dir / "annotated.mp4",
            alerts_dir=Path("outputs/alerts") / f"{video_stem}_{mode}",
            alerts_log_path=run_dir / "alerts.jsonl",
        )

        alert_frames = [a.frame_index for a in detector.alerts]
        last_frame = detector.frame_count - 1

        if alert_frames:
            first_alert, last_alert = alert_frames[0], alert_frames[-1]
            # "recovered" = similarity climbed back above threshold for the
            # last few frames, i.e. the alert isn't still firing at the end.
            recovered = last_alert < last_frame - 2
            note = f"{first_alert:>11}  {last_alert:>10}  {'yes' if recovered else 'no'}"
        else:
            note = f"{'-':>11}  {'-':>10}  n/a (never alerted)"

        print(
            f"{video_stem:<18}{mode:<9}{detector.frame_count:>7}{len(alert_frames):>8}"
            f"{detector.average_similarity:>9.4f}   {note}"
        )

    print(f"\nAnnotated videos + alert logs written under {OUTPUT_DIR}/")
    print("Flagged-frame JPEGs written under outputs/alerts/<video>_<mode>/")


if __name__ == "__main__":
    main()

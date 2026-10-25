"""Check RetailS split structure and all test masks; sample real pose schemas."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.data.retails import SUBSET_DIRS, discover_retails_root, find_pose_files, gt_dir_for_subset, label_path_for_pose_file, load_pose_file


def main():
    root = discover_retails_root(ROOT / "data/retails")
    if root is None:
        raise SystemExit("RetailS_train not found under data/retails")
    report = {"root": str(root), "official_source": "https://github.com/TeCSAR-UNCC/RetailS#dataset-download-instructions",
              "download_url": "https://drive.google.com/file/d/1uBCvDm7QdYjxwS9HT63lYaucATrecNuH/view",
              "subsets": {}, "scope": "All test-mask associations, values and test pose-frame bounds; pose coordinate schemas sampled from first 5 files per subset."}
    for key, directory in SUBSET_DIRS.items():
        subset = root / directory
        files = find_pose_files(subset)
        if not files:
            raise ValueError(f"No pose files for {key}")
        counts = {"pose_files": len(files), "label_files_verified": 0, "normal_frames": 0, "anomaly_frames": 0}
        gt = gt_dir_for_subset(subset)
        if key != "train":
            counts["test_pose_frame_ids_verified"] = 0
            for path in files:
                mask_path = label_path_for_pose_file(path, gt)
                if mask_path is None or not mask_path.is_file():
                    raise FileNotFoundError(f"Missing test mask: {path.name}")
                labels = np.load(mask_path, allow_pickle=False)
                if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
                    raise ValueError(f"Expected 1D binary test mask: {mask_path}")
                counts["label_files_verified"] += 1
                counts["normal_frames"] += int((labels == 0).sum())
                counts["anomaly_frames"] += int((labels == 1).sum())
                pose_records = json.loads(path.read_text(encoding="utf-8"))
                for observations in pose_records.values():
                    frame_ids = [int(fid) for fid in observations]
                    if any(fid < 0 or fid >= len(labels) for fid in frame_ids):
                        raise ValueError(f"Pose frame ID outside its test mask: {path}")
                    counts["test_pose_frame_ids_verified"] += len(frame_ids)
        sampled_observations = 0
        nan_confidence = 0
        above_one = 0
        for path in files[:5]:
            labels = None if key == "train" else np.load(label_path_for_pose_file(path, gt), allow_pickle=False)
            for track in load_pose_file(path):
                for fid, frame in track.frames.items():
                    points = frame.keypoints
                    if points.shape != (17, 3) or not np.isfinite(points[:, :2]).all():
                        raise ValueError(f"Malformed/non-finite sampled pose: {path}, frame {fid}")
                    conf = points[:, 2]
                    nan_confidence += int(np.isnan(conf).sum())
                    valid_conf = conf[~np.isnan(conf)]
                    # RetailS HRNet heatmap scores are not calibrated sigmoid
                    # probabilities; real observations can be slightly > 1.
                    above_one += int((valid_conf > 1).sum())
                    if not np.isfinite(valid_conf).all() or (valid_conf < 0).any():
                        raise ValueError(f"Invalid sampled confidence: {path}, frame {fid}")
                    if labels is not None and not 0 <= fid < len(labels):
                        raise ValueError(f"Sampled frame ID outside test mask: {path}, {fid}")
                    sampled_observations += 1
        counts.update(sampled_pose_files=min(5, len(files)), sampled_person_frame_observations=sampled_observations,
                      sampled_nan_confidences=nan_confidence)
        counts["sampled_confidences_above_one"] = above_one
        report["subsets"][key] = counts
        print(key, counts, flush=True)
    report["passed"] = True
    out = ROOT / "outputs/logs/retails_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PASS: RetailS structure, test masks, and sampled schemas verified.", out)


if __name__ == "__main__":
    main()

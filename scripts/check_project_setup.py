"""Read-only development readiness check; missing later-stage assets are reported separately."""

import importlib.util
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = {"python": sys.version.split()[0], "executable": sys.executable,
              "dependencies": {name: importlib.util.find_spec(name) is not None
                               for name in ["cv2", "numpy", "torch", "ultralytics", "lap"]},
              "classes": {}}
    valid = True
    for label in ["shoplifting", "normal"]:
        videos = []
        for path in sorted((ROOT / "data/external_video/videos" / label).glob("*.mp4")):
            cap = cv2.VideoCapture(str(path))
            count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
            ok, _ = cap.read()
            cap.release()
            readable = bool(ok and count > 0 and fps > 0)
            valid &= readable
            videos.append({"name": path.name, "frames": count, "fps": fps, "readable": readable})
        valid &= bool(videos)
        report["classes"][label] = {"videos": videos,
            "sampled_images": len(list((ROOT / "data/external_video/frames" / label).glob("*.jpg")))}
    report["later_stage_assets"] = {
        "retails_pose_json_count": sum(1 for _ in (ROOT / "data/retails").rglob("*.json")),
        "motionbert_code": (ROOT / "third_party/MotionBERT/lib/model/DSTformer.py").exists(),
        "motionbert_weights": (ROOT / "third_party/MotionBERT/checkpoint/pose3d/MB_ft_h36m/best_epoch.bin").exists(),
        "experiment_d_checkpoint": (ROOT / "outputs/checkpoints/experiment_d/model.pt").exists(),
        "yolo_pose_weights": (ROOT / "third_party/yolo/yolov8n-pose.pt").exists(),
    }
    valid &= all(report["dependencies"].values())
    report["ready_for_2d_pose_development"] = bool(valid)
    print(json.dumps(report, indent=2))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

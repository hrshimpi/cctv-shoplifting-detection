"""Regression checks that need no network or pretrained model."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

from scripts.extract_video_poses import validate_poses
from src.data.preprocessing import build_windows
from src.demo.video_pose import detect_and_track


class VideoPoseTests(unittest.TestCase):
    def test_source_fps_and_gpu_compatible_keypoint_conversion(self):
        # Model timings do not contain the source FPS. Preserve 12 FPS,
        # rather than the old 25-FPS fallback, for downstream rendering.
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "sample.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 12, (64, 48))
            self.assertTrue(writer.isOpened())
            for _ in range(4):
                writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.release()
            result = SimpleNamespace(boxes=SimpleNamespace(id=torch.tensor([7.])),
                                     keypoints=SimpleNamespace(data=torch.ones((1, 17, 3))))
            seen = {}
            def fake_track(**kwargs):
                seen.update(kwargs)
                return iter([result] * 4)
            model = SimpleNamespace(track=fake_track)
            with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=lambda _: model)}):
                tracks, fps, size = detect_and_track(video, img_size=960)
            self.assertAlmostEqual(fps, 12)
            self.assertEqual(size, (64, 48))
            self.assertEqual(tracks[7].frame_ids_sorted, [0, 1, 2, 3])
            self.assertEqual(tracks[7].frames[0].keypoints.shape, (17, 3))
            self.assertEqual(seen["imgsz"], 960)

    def test_missing_video_is_reported_before_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                detect_and_track(Path(tmp) / "missing.mp4")

    def test_export_validation_rejects_nonfinite_and_out_of_range_frames(self):
        points = [1., 2., .9] * 17
        poses = {"1": {"0": {"keypoints": points}}}
        self.assertEqual(validate_poses(poses, 10)["frames_with_tracks"], 1)
        with self.assertRaises(ValueError):
            validate_poses({"1": {"10": {"keypoints": points}}}, 10)
        bad = points.copy()
        bad[0] = float("nan")
        with self.assertRaises(ValueError):
            validate_poses({"1": {"0": {"keypoints": bad}}}, 10)
        with self.assertRaises(ValueError):
            validate_poses({}, 10)

    def test_temporal_windows_do_not_bridge_tracking_gaps(self):
        frame_ids = np.concatenate([np.arange(30), np.arange(40, 70)])
        windows = build_windows(frame_ids, np.ones((60, 17, 3)),
                                np.ones((60, 17), dtype=bool), window_length=30, stride=15)
        self.assertEqual([(w["start_frame"], w["end_frame"]) for w in windows], [(0, 29), (40, 69)])
        self.assertTrue(all(w["keypoints"].shape == (30, 17, 3) for w in windows))


if __name__ == "__main__":
    unittest.main()

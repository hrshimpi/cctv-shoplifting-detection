import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.external_sequences import build_sequence_arrays, export_sequences, validate_sequence_arrays, verify_sequences
from src.data.preprocessing import preprocess_track
from src.data.retails import PersonTrack, PoseFrame

CFG = {"confidence_threshold": 0.3, "max_interpolation_gap": 5, "smoothing_window": 3,
       "window_length": 30, "stride": 15}


def records(confidence=1.):
    result = {}
    for fid in [*range(30), *range(90, 120)]:
        points = np.array([[j * 2. + fid, j * 3., confidence] for j in range(17)], dtype=np.float32)
        result[str(fid)] = {"keypoints": points.reshape(-1).tolist(), "scores": None}
    return {"7": result}


class ExternalSequenceTests(unittest.TestCase):
    def test_smoothing_does_not_mix_disconnected_track_segments(self):
        frames = {}
        for fid in [*range(30), *range(90, 120)]:
            points = np.ones((17, 3), dtype=np.float32)
            points[:, :2] = 0. if fid < 30 else 100.
            frames[fid] = PoseFrame(fid, points, None)
        track = PersonTrack("7", "demo", "clip", "1", Path("clip.mp4"), frames)
        processed = preprocess_track(track, smoothing_window=3, apply_normalization=False)
        np.testing.assert_array_equal(processed["keypoints"][:30, :, :2], 0.)
        np.testing.assert_array_equal(processed["keypoints"][30:, :, :2], 100.)

    def test_windows_preserve_track_frame_ids_and_features(self):
        arrays, rows = build_sequence_arrays(records(), Path("clip.mp4"), CFG)
        self.assertEqual(validate_sequence_arrays(arrays, 120, CFG), 2)
        self.assertEqual(arrays["frame_ids"][:, 0].tolist(), [0, 90])
        self.assertEqual(arrays["frame_ids"][:, -1].tolist(), [29, 119])
        self.assertEqual(arrays["track_ids"].tolist(), ["7", "7"])
        self.assertEqual(arrays["features_a"].shape, (2, 30, 34))
        self.assertEqual(arrays["features_b"].shape, (2, 30, 58))
        self.assertEqual(rows[0]["candidate_windows"], 2)

    def test_low_visibility_is_rejected_without_inventing_labels(self):
        arrays, rows = build_sequence_arrays(records(0.1), Path("clip.mp4"), CFG)
        self.assertEqual(validate_sequence_arrays(arrays, 120, CFG), 0)
        self.assertEqual(rows[0]["rejected_windows"], 2)
        self.assertNotIn("labels", arrays)
        self.assertNotIn("y", arrays)

    def test_modified_features_and_frame_gaps_fail_validation(self):
        arrays, _ = build_sequence_arrays(records(), Path("clip.mp4"), CFG)
        arrays["features_a"][0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "aligned"):
            validate_sequence_arrays(arrays, 120, CFG)
        arrays, _ = build_sequence_arrays(records(), Path("clip.mp4"), CFG)
        arrays["frame_ids"][0, 15] += 1
        with self.assertRaisesRegex(ValueError, "gap"):
            validate_sequence_arrays(arrays, 120, CFG)

    def test_visible_limbs_do_not_replace_missing_normalization_anchors(self):
        data = records()
        for record in data["7"].values():
            for joint in [11, 12]:
                record["keypoints"][joint * 3 + 2] = 0.
        arrays, rows = build_sequence_arrays(data, Path("clip.mp4"), CFG)
        self.assertEqual(len(arrays["keypoints"]), 0)
        self.assertEqual(rows[0]["rejected_windows"], 2)

    def test_round_trip_and_changed_source_detection(self):
        test_root = Path(__file__).resolve().parents[1] / "outputs/test_tmp"
        test_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as tmp:
            root = Path(tmp)
            source = root / "clip.mp4"
            source.write_bytes(b"fixture-source-identity")
            pose = root / "poses"
            pose.mkdir()
            (pose / "poses.json").write_text(json.dumps(records()))
            (pose / "summary.json").write_text(json.dumps({"source_video": str(source), "source_frames": 120,
                                                          "fps": 30., "confidence_threshold": 0.5}))
            manifest = export_sequences(pose, root / "export", "normal", CFG)
            self.assertEqual(manifest["retained_windows"], 2)
            self.assertEqual(verify_sequences(root / "export")["label_scope"], "video only; no per-window anomaly ground truth")
            source.write_bytes(b"different-source")
            with self.assertRaisesRegex(ValueError, "Source video changed"):
                verify_sequences(root / "export")


if __name__ == "__main__":
    unittest.main()

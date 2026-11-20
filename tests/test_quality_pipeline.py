import unittest
from pathlib import Path
import numpy as np
import torch
from src.data.quality import prepare_track, feature_mask, h36m_visibility, masked_error
from src.data.retails import PersonTrack, PoseFrame

CFG = dict(coordinate_order='yx', confidence_threshold=.3, max_interpolation_gap=0,
           smoothing_window=3, max_normalized_coordinate=10, min_visible_fraction=.5)


class QualityPipelineTests(unittest.TestCase):
    def track(self, points):
        return PersonTrack('1', '1', '1', '1', Path('source.json'),
                           {i: PoseFrame(i, p, None) for i, p in enumerate(points)})

    def points(self):
        p = np.ones((5, 17, 3), dtype=np.float32)
        p[..., 0], p[..., 1] = 100, 200
        p[:, [5, 6], 0] = 50
        p[:, [11, 12], 0] = 100
        return p

    def test_axis_conversion_and_missing_values_never_enter_features(self):
        p = self.points()
        p[:, 9] = [1e8, -1e8, 0]
        result = prepare_track(self.track(p), CFG)
        self.assertTrue(result['anchor_valid'].all())
        self.assertTrue((result['keypoints'][:, [5, 6], 1] == -1).all())
        self.assertTrue((result['keypoints'][:, 9] == 0).all())
        self.assertLessEqual(np.abs(result['keypoints'][..., :2]).max(), 1)

    def test_missing_hip_invalidates_normalization(self):
        p = self.points()
        p[:, 11, 2] = 0
        self.assertFalse(prepare_track(self.track(p), CFG)['anchor_valid'].any())

    def test_velocity_and_derived_masks_require_all_support(self):
        visible = np.ones((4, 17), dtype=bool)
        visible[1, 9] = False
        mask = feature_mask(visible, 'b')
        self.assertEqual(mask.shape, (4, 58))
        self.assertFalse(mask[0, 34:].any())
        # Third motion joint is the left elbow, fifth is the left wrist.
        self.assertFalse(mask[1:3, 34 + 4 * 2:34 + 5 * 2].any())
        visible[:, 11] = False
        self.assertFalse(h36m_visibility(visible)[:, 0].any())

    def test_masked_loss_ignores_missing_targets_and_rejects_empty_support(self):
        target = torch.tensor([[[1., 1000.]]])
        prediction = torch.zeros_like(target, requires_grad=True)
        mask = torch.tensor([[[True, False]]])
        error = masked_error(prediction, target, mask)
        self.assertEqual(error.item(), 1.)
        error.sum().backward()
        self.assertEqual(prediction.grad[0, 0, 1].item(), 0.)
        with self.assertRaises(ValueError):
            masked_error(prediction, target, torch.zeros_like(mask))


if __name__ == '__main__':
    unittest.main()

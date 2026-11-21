import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.data.clip_classification import clip_descriptor, verify_split, portable_probability
from scripts.prepare_mnnit_clips import safe_member, sample_indices


class ClipClassificationTests(unittest.TestCase):
    def test_descriptor_preserves_mean_and_changes(self):
        values=np.array([[1,2],[3,4],[5,6]],dtype=np.float32)
        result=clip_descriptor(values)
        np.testing.assert_allclose(result[:2],[3,4])
        np.testing.assert_allclose(result[-2:],[2,2])
        self.assertEqual(clip_descriptor(values,False).shape,(4,))
        np.testing.assert_array_equal(clip_descriptor(np.ones((3,2)))[-2:],[0,0])

    def test_invalid_descriptor_input_is_rejected(self):
        for value in [np.ones(3),np.ones((1,3)),np.array([[1,np.nan],[2,3]])]:
            with self.assertRaises(ValueError): clip_descriptor(value)

    def test_duplicate_groups_cannot_cross_splits(self):
        rows=[]
        for split in ['train','validation','test']:
            for label in [0,1]:
                token=split+str(label)
                rows.append(dict(split=split,label=label,clip_id=token,source_sha256=token,sampled_sha256=token,duplicate_group=token))
        verify_split(rows)
        for key in ['source_sha256','sampled_sha256','duplicate_group','clip_id']:
            changed=[dict(r) for r in rows];changed[-1][key]=changed[0][key]
            with self.assertRaises(ValueError): verify_split(changed)
        with self.assertRaises(ValueError): verify_split(rows[:-1])

    def test_portable_classifier_stable_for_extreme_values(self):
        probabilities=portable_probability(np.array([[-10000.],[0.],[10000.]]),np.array([0.]),np.array([1.]),np.array([1.]),np.array([0.]))
        np.testing.assert_allclose(probabilities,[0,.5,1])
        with self.assertRaises(ValueError): portable_probability(np.array([[1.]]),[0],[0],[1],[0])

    def test_zip_paths_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.assertEqual(safe_member(root,'Dataset/Normal/a.mp4'),(root/'Dataset/Normal/a.mp4').resolve())
            for name in ['../../escape.mp4','..\\escape.mp4','C:/escape.mp4','/escape.mp4']:
                with self.assertRaises(ValueError): safe_member(root,name)

    def test_sampling_retains_endpoints_without_inventing_more_source_frames(self):
        ids=sample_indices(3)
        self.assertEqual(len(ids),16)
        self.assertEqual((ids[0],ids[-1]),(0,2))
        self.assertTrue((np.diff(ids)>=0).all())
        with self.assertRaises(ValueError): sample_indices(1)

"""Unit tests for research_v2 (folds, frame protocol, metrics, scoring, windowing, augmentation, flow).

These use small hand-built arrays to check the code; they produce no reported metrics."""
import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from src.data.preprocessing import build_windows
from src.data.quality import accepted_window
from src.evaluation import folds as F
from src.evaluation.frame_protocol import flatten, frame_scores
from src.evaluation.metrics import (balanced_accuracy, detection_metrics, fast_roc_auc, grouped_bootstrap,
                                    recall_at_fpr, specificity_at_recall)
from src.evaluation.v2_scoring import Calibrator, window_score
from src.experiments.v2_data import window_starts
from src.experiments.v2_train import FLIP_INDEX, augment
from src.models.stg_nf import STGNF, window_normalise


# ---- step 0: folds --------------------------------------------------------------------------

def _videos():
    rng = np.random.default_rng(1)
    vids = []
    for subset in ('test_realworld', 'test_staged'):
        for cam in range(1, 7):
            for i in range(int(rng.integers(2, 9))):
                anomaly = int(rng.integers(0, 300)) if i % 3 else 0
                vids.append(dict(subset=subset, camera=cam, stem=f'{cam}_{subset[5]}{i}', normal=int(rng.integers(1, 500)),
                                 anomaly=anomaly))
    vids.append(dict(subset='test_staged', camera=9, stem='9_lonely', normal=10, anomaly=5))   # single-video group
    return vids


def test_folds_disjoint_cover_and_stratified():
    vids = _videos()
    a = F.make_folds(vids, seed=0)
    assert set(a) == {F.video_key(v['subset'], v['camera'], v['stem']) for v in vids}
    assert set(a.values()) <= {'A', 'B'}                       # each video in exactly one fold
    assert F.check_folds(vids, a) == []
    for subset in ('test_realworld', 'test_staged'):
        for cam in range(1, 7):                                  # every camera represented in both folds
            got = {a[F.video_key(v['subset'], v['camera'], v['stem'])] for v in vids if (v['subset'], v['camera']) == (subset, cam)}
            assert got == {'A', 'B'}


def test_folds_are_deterministic():
    vids = _videos()
    assert F.make_folds(vids, 0) == F.make_folds(vids, 0)


def test_fold_repair_moves_missing_class():
    vids = [dict(subset='s', camera=1, stem=f'v{i}', normal=100, anomaly=0) for i in range(3)]
    vids.append(dict(subset='s', camera=1, stem='pos1', normal=5, anomaly=50))
    vids.append(dict(subset='s', camera=1, stem='pos2', normal=5, anomaly=40))
    a = F.make_folds(vids, 3)
    assert F.check_folds(vids, a) == []


# ---- step 0b: frame protocol ----------------------------------------------------------------

def test_frame_protocol_two_people_ten_frames():
    # person p1 (track 0): windows [0,3] score 1 and [2,5] score 3 -> frames 0-1:1, 2-5:3
    # person p2 (track 1): window [4,7] score 2 -> frames 4-5: max(3,2)=3, 6-7: 2
    # frames 8-9 unscored -> filled with the subset minimum (1)
    out = frame_scores([1., 3., 2.], ['v', 'v', 'v'], [0, 0, 1], [0, 2, 4], [3, 5, 7], lengths={'v': 10})
    expected = np.array([1, 1, 3, 3, 3, 3, 2, 2, np.nan, np.nan])
    np.testing.assert_array_equal(np.isnan(out['v']), np.isnan(expected))
    np.testing.assert_allclose(out['v'][:8], expected[:8])
    flat = flatten(out, {'v': np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 0])}, {'v': 3})
    np.testing.assert_allclose(flat['scores'], [1, 1, 3, 3, 3, 3, 2, 2, 1, 1])
    assert flat['scored'].sum() == 8 and (flat['camera'] == 3).all()


def test_frame_protocol_includes_videos_without_windows():
    out = frame_scores([1.], ['a'], [0], [0], [1], lengths={'a': 3, 'b': 4})
    assert np.isnan(out['b']).all() and len(out['b']) == 4


def test_smoothing_never_crosses_track_gaps():
    # one track, two runs separated by a gap (frames 4-5 uncovered)
    out = frame_scores([0., 10.], ['v', 'v'], [0, 0], [0, 6], [3, 9], lengths={'v': 10}, sigma=2.0)
    np.testing.assert_allclose(out['v'][:4], 0.0)               # first run untouched by the 10s
    np.testing.assert_allclose(out['v'][6:], 10.0)
    assert np.isnan(out['v'][4:6]).all()


def test_frame_protocol_rejects_bad_input():
    with pytest.raises(ValueError):
        frame_scores([np.nan], ['v'], [0], [0], [1])
    with pytest.raises(ValueError):
        frame_scores([1.], ['v'], [0], [0], [5], lengths={'v': 3})


# ---- step 1: metrics ------------------------------------------------------------------------

def test_fast_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.normal(size=500) + y, 1)
    assert fast_roc_auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert np.isnan(fast_roc_auc(np.ones(5), np.arange(5)))


def test_balanced_and_operating_point_metrics():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0, 1, 2, 5, 3, 6, 7, 8.])
    assert balanced_accuracy(y, s, 3) == pytest.approx((3 / 4 + 4 / 4) / 2)
    assert recall_at_fpr(y, s, 0.0) == pytest.approx(0.75)      # threshold 6 -> 3/4 positives, no FP
    assert specificity_at_recall(y, s, 1.0) == pytest.approx(0.75)


def test_grouped_bootstrap_resamples_groups_and_is_reproducible():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 50)
    y = (rng.random(1000) < 0.4).astype(int)
    s = y + rng.normal(size=1000)
    lo, hi, a = grouped_bootstrap(y, s, groups, draws=200, seed=0)
    _, _, b = grouped_bootstrap(y, s, groups, draws=200, seed=0)
    np.testing.assert_array_equal(a, b)
    assert lo < fast_roc_auc(y, s) < hi


def test_detection_metrics_fields():
    y = np.array([0, 1, 0, 1, 1, 0])
    m = detection_metrics(y, np.array([.1, .9, .2, .8, .4, .3]), 0.5, cameras=np.array([1, 1, 1, 2, 2, 2]),
                          groups=np.array([0, 0, 1, 1, 2, 2]), draws=50)
    for key in ('roc_auc', 'pr_auc', 'balanced_accuracy', 'recall_at_fpr5', 'specificity_at_recall90',
                'camera_auc_mean', 'camera_auc_std', 'roc_auc_ci', 'base_rate', 'pr_auc_lift'):
        assert key in m
    assert m['roc_auc'] == 1.0 and m['per_camera_auc'][1] == 1.0


# ---- step 2: scoring variants + calibration -------------------------------------------------

def test_window_score_variants():
    err = np.zeros((1, 4, 17), np.float32)
    mask = np.ones((1, 4, 17), bool)
    err[0, 0, 9] = 17 * 4                                        # one big left-wrist error in frame 0
    assert window_score(err, mask, 'mean')[0] == pytest.approx(1.0)
    assert window_score(err, mask, 'max_joint')[0] == pytest.approx(17.0)
    assert window_score(err, mask, 'topk3')[0] == pytest.approx(4 / 3)
    assert window_score(err, mask, 'upper_body')[0] == pytest.approx(68 / 24)
    assert window_score(err, mask, 'no_face')[0] == pytest.approx(68 / 48)
    mask[0, :, 9] = False                                        # masked joints never count
    assert window_score(err, mask, 'max_joint')[0] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        window_score(err, mask, 'nope')


def test_upper_body_falls_back_when_unsupported():
    err = np.ones((1, 2, 17), np.float32)
    mask = np.zeros((1, 2, 17), bool)
    mask[0, :, 0] = True
    assert window_score(err, mask, 'upper_body')[0] == pytest.approx(1.0)


def test_calibrator_per_camera_uses_only_its_camera():
    rng = np.random.default_rng(0)
    val = np.concatenate([rng.lognormal(0, .1, 200), rng.lognormal(2, .1, 200)])
    cams = np.repeat([1, 2], 200)
    cal = Calibrator(log=True, camnorm=True).fit(val, cams)
    z = cal.transform(val, cams)
    assert abs(z[:200].mean()) < 1e-6 and abs(z[200:].mean()) < 1e-6
    pooled = Calibrator(log=True, camnorm=False).fit(val, cams).transform(val, cams)
    assert pooled[200:].mean() > 0.5                             # camera 2 looks anomalous without camnorm


# ---- windowing matches research_v1 ----------------------------------------------------------

def test_window_starts_match_research_v1_rules():
    rng = np.random.default_rng(0)
    frame_ids = np.concatenate([np.arange(0, 70), np.arange(80, 150)])
    vis = rng.random((140, 17)) > 0.35
    anchor = rng.random(140) > 0.02
    cfg = dict(min_visible_fraction=0.5)
    expected = []
    for w in build_windows(frame_ids, np.zeros((140, 17, 3)), vis, 30, 15):
        start = int(np.searchsorted(frame_ids, w['start_frame']))
        if accepted_window(w, anchor[start:start + 30], cfg):
            expected.append(start)
    got = window_starts(frame_ids, anchor, vis, 30, 15, 0.5)
    np.testing.assert_array_equal(got, expected)


# ---- step 4: augmentation -------------------------------------------------------------------

def test_flip_swaps_left_right_and_negates_x():
    xy = torch.zeros(1, 2, 17, 2)
    xy[0, :, 9] = torch.tensor([1.0, 2.0])                        # left wrist
    vis = torch.ones(1, 2, 17, dtype=torch.bool)
    vis[0, :, 3] = False                                          # left ear missing
    g = torch.Generator().manual_seed(0)
    x_in, x_t, v_in, v_t = augment(xy, vis, dict(flip=1.0), g)
    np.testing.assert_allclose(x_t[0, 0, 10].numpy(), [-1.0, 2.0])  # now the right wrist
    assert not v_t[0, 0, 4] and v_t[0, 0, 3]
    assert FLIP_INDEX[FLIP_INDEX[5]] == 5


def test_joint_dropout_only_hits_the_input():
    xy = torch.randn(8, 10, 17, 2)
    vis = torch.ones(8, 10, 17, dtype=torch.bool)
    g = torch.Generator().manual_seed(0)
    x_in, x_t, v_in, v_t = augment(xy, vis, dict(joint_dropout=0.5), g)
    assert v_t.all() and 0.3 < v_in.float().mean() < 0.7
    assert torch.all(x_in[~v_in] == 0)
    torch.testing.assert_close(x_t, xy)


# ---- step 6: flow ---------------------------------------------------------------------------

def test_flow_logdet_matches_autograd_jacobian():
    torch.manual_seed(0)
    model = STGNF(2, hidden=4, steps=2).double()
    x = torch.randn(4, 2, 3, 17, dtype=torch.float64)
    model.train()
    model(x)                                                      # ActNorm data init
    with torch.no_grad():                                         # make couplings non-trivial
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    model.eval()
    one = x[:1]
    _, logdet = model(one)
    jac = torch.autograd.functional.jacobian(lambda v: model(v.reshape(1, 2, 3, 17))[0].flatten(), one.flatten())
    assert float(logdet[0].detach()) == pytest.approx(float(torch.slogdet(jac)[1]), rel=1e-5)  # log-det kept in fp32


def test_flow_nll_and_window_normalise():
    xy = torch.randn(5, 24, 17, 2)
    vis = torch.rand(5, 24, 17) > 0.2
    x = window_normalise(xy, vis)
    assert x.shape == (5, 2, 24, 17) and torch.all(x.permute(0, 2, 3, 1)[~vis] == 0)
    model = STGNF(2, hidden=8, steps=2)
    nll = model.nll(x)
    assert nll.shape == (5,) and torch.isfinite(nll).all()
    nll.mean().backward()

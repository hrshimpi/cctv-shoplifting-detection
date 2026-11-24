import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data import video_clip_features as vcf


@pytest.mark.parametrize('h,w', [(480, 640), (1080, 1920), (1920, 1080), (500, 500), (240, 320), (37, 301)])
def test_letterbox_keeps_aspect_and_pads(h, w):
    frame = np.full((h, w, 3), 200, np.uint8)
    out = vcf.letterbox(frame, 224)
    assert out.shape == (224, 224, 3)
    rows = np.where(out[:, :, 0].max(1) > 0)[0]; cols = np.where(out[:, :, 0].max(0) > 0)[0]
    ch, cw = np.ptp(rows) + 1, np.ptp(cols) + 1
    assert max(ch, cw) == 224
    assert abs(cw / ch - w / h) / (w / h) < 0.05 or min(ch, cw) <= 2


def test_indices_and_segments():
    f = vcf.frame_indices(0, 100); assert len(f) == vcf.FRAME_SAMPLES and f[0] == 0 and f[-1] == 99
    win = vcf.window_indices(0, 10)  # shorter than one window: clamped, still valid
    assert win.shape == (vcf.VIDEO_WINDOWS, vcf.VIDEO_WINDOW_FRAMES) and win.min() >= 0 and win.max() <= 9
    assert vcf.segments_for(300, 30) == [(0, 300)]
    segs = vcf.segments_for(1000, 30, 10)
    assert segs[0] == (0, 300) and segs[-1][1] == 1000 and all(b[0] == a[1] for a, b in zip(segs, segs[1:]))


def test_descriptors_shapes():
    d = vcf.descriptors(np.random.rand(32, 1280), np.random.rand(4, 512))
    assert d['frame'].shape == (3840,) and d['video'].shape == (1024,) and d['frame_video'].shape == (4864,)


@pytest.mark.parametrize('pca', [0, 5])
def test_portable_head_matches_sklearn(pca):
    rng = np.random.default_rng(0); x = rng.normal(size=(60, 20)); y = (x[:, 0] + rng.normal(size=60) > 0).astype(int)
    s = StandardScaler().fit(x); z = s.transform(x)
    p = PCA(pca, random_state=0).fit(z) if pca else None
    z2 = p.transform(z) if p else z
    clf = LogisticRegression(C=0.5).fit(z2, y)
    head = dict(mean=s.mean_, scale=s.scale_, pca_mean=p.mean_ if p else np.zeros(0),
                pca_components=p.components_ if p else np.zeros((0, 0)), coefficient=clf.coef_, intercept=clf.intercept_)
    np.testing.assert_allclose(vcf.head_probability(x, head), clf.predict_proba(z2)[:, 1], atol=1e-10)

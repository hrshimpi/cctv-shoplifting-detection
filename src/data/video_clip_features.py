"""Aspect-ratio-safe frame sampling and frozen-backbone clip features (MNNIT v2).

Used identically by training (scripts/train_mnnit_clip_v2.py) and by single-video
inference (scripts/predict_mnnit_clip.py), so an uploaded 16:9, 9:16, 4:3 or 1:1
video is preprocessed exactly like the training clips: frames are letterboxed
(scaled to fit, then padded) into a square, never stretched.
"""
import cv2
import numpy as np

FRAME_SIZE = 224          # ImageNet frame backbone input
VIDEO_SIZE = 112          # Kinetics video backbone input
FRAME_SAMPLES = 32        # uniformly spaced frames per clip/chunk for the frame backbone
VIDEO_WINDOWS = 4         # short windows per clip/chunk for the video backbone
VIDEO_WINDOW_FRAMES = 16  # frames per video window
VIDEO_STRIDE = 2          # frame stride inside a window (16 x 2 = ~1 s at 30 FPS)
PAD_VALUE = 0             # black padding


def letterbox(frame, size, pad_value=PAD_VALUE):
    """Resize keeping aspect ratio so the longer side equals `size`, pad to size x size."""
    if frame is None or frame.ndim != 3 or min(frame.shape[:2]) < 1:
        raise ValueError('Expected an HxWx3 frame')
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    out = np.full((size, size, 3), pad_value, dtype=frame.dtype)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out


def frame_indices(start, end, samples=FRAME_SAMPLES):
    """Uniformly spaced indices in [start, end)."""
    if end - start < 2:
        raise ValueError('Segment must have at least two frames')
    return np.linspace(start, end - 1, samples).round().astype(np.int64)


def window_indices(start, end, windows=VIDEO_WINDOWS, length=VIDEO_WINDOW_FRAMES, stride=VIDEO_STRIDE):
    """`windows` evenly placed short windows; each is `length` frames at `stride`, clamped to the segment."""
    span = (length - 1) * stride
    last_start = max(start, end - 1 - span)
    starts = np.linspace(start, last_start, windows).round().astype(np.int64)
    idx = starts[:, None] + np.arange(length)[None, :] * stride
    return np.clip(idx, start, end - 1)


def count_frames(path):
    """Exact decodable frame count and FPS (container metadata can be wrong)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    if n < 2:
        raise ValueError(f'Video has fewer than two decodable frames: {path}')
    return n, (fps if np.isfinite(fps) and fps > 0 else 30.0)


def segments_for(total, fps, chunk_seconds=None):
    """Whole clip, or consecutive chunks of ~chunk_seconds (last short chunk merged)."""
    if not chunk_seconds:
        return [(0, total)]
    size = max(2, int(round(chunk_seconds * fps)))
    bounds = list(range(0, total, size))
    segs = [(s, min(s + size, total)) for s in bounds]
    if len(segs) > 1 and segs[-1][1] - segs[-1][0] < size // 2:
        segs[-2] = (segs[-2][0], segs[-1][1]); segs.pop()
    return segs


def read_letterboxed(path, segments):
    """One sequential decode. Returns per segment dict(frames=[N,224,224,3] RGB, windows=[W,L,112,112,3] RGB)."""
    plans = [(frame_indices(s, e), window_indices(s, e)) for s, e in segments]
    want_f = {int(i) for f, _ in plans for i in f}
    want_v = {int(i) for _, v in plans for i in v.ravel()}
    big, small = {}, {}
    cap = cv2.VideoCapture(str(path))
    fid = 0
    last = max(want_f | want_v)
    while fid <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if fid in want_f or fid in want_v:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if fid in want_f:
                big[fid] = letterbox(rgb, FRAME_SIZE)
            if fid in want_v:
                small[fid] = letterbox(rgb, VIDEO_SIZE)
        fid += 1
    cap.release()
    missing = (want_f - big.keys()) | (want_v - small.keys())
    if missing:
        raise ValueError(f'Could not decode {len(missing)} requested frames from {path}')
    return [dict(frames=np.stack([big[int(i)] for i in f]),
                 windows=np.stack([np.stack([small[int(i)] for i in row]) for row in v]))
            for f, v in plans]


class Backbones:
    """Frozen pretrained encoders: EfficientNet-B0 (ImageNet) per frame, R(2+1)D-18 (Kinetics-400) per window."""

    def __init__(self, device=None, pretrained=True):
        import torch
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        fw = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        vw = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.frame = efficientnet_b0(weights=fw); self.frame.classifier = torch.nn.Identity()
        self.video = r2plus1d_18(weights=vw); self.video.fc = torch.nn.Identity()
        for m in (self.frame, self.video):
            m.eval().to(self.device)
            for p in m.parameters():
                p.requires_grad_(False)
        # Normalization constants published with each weight set.
        self.f_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.f_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.v_mean = torch.tensor([0.43216, 0.394666, 0.37645], device=self.device).view(1, 3, 1, 1, 1)
        self.v_std = torch.tensor([0.22803, 0.22145, 0.216989], device=self.device).view(1, 3, 1, 1, 1)
        self.names = dict(frame='efficientnet_b0 IMAGENET1K_V1' if pretrained else 'efficientnet_b0 RANDOM',
                          video='r2plus1d_18 KINETICS400_V1' if pretrained else 'r2plus1d_18 RANDOM')

    def embed(self, segment):
        """Returns (frame_embeddings [N,1280], window_embeddings [W,512]) as float32 numpy."""
        t = self.torch
        with t.inference_mode():
            x = t.from_numpy(segment['frames']).to(self.device).permute(0, 3, 1, 2).float() / 255
            f = self.frame((x - self.f_mean) / self.f_std).float().cpu().numpy()
            v = t.from_numpy(segment['windows']).to(self.device).permute(0, 4, 1, 2, 3).float() / 255  # W,C,T,H,W
            w = self.video((v - self.v_mean) / self.v_std).float().cpu().numpy()
        return f.astype(np.float32), w.astype(np.float32)


def descriptors(frame_emb, window_emb):
    """Named clip descriptors from raw embeddings (feature sets compared by cross-validation)."""
    f = np.asarray(frame_emb); w = np.asarray(window_emb)
    if f.ndim != 2 or len(f) < 2 or w.ndim != 2 or not (np.isfinite(f).all() and np.isfinite(w).all()):
        raise ValueError('Finite frame (>=2) and window embeddings are required')
    frame = np.concatenate([f.mean(0), f.std(0), np.abs(np.diff(f, axis=0)).mean(0)])
    video = np.concatenate([w.mean(0), w.max(0)])
    return dict(frame=frame.astype(np.float32), video=video.astype(np.float32),
                frame_video=np.concatenate([frame, video]).astype(np.float32))


def head_probability(x, head):
    """Portable scaler -> optional PCA -> logistic head, from NPZ arrays (no pickle)."""
    z = (np.asarray(x, dtype=np.float64) - head['mean']) / head['scale']
    if head['pca_components'].size:
        z = (z - head['pca_mean']) @ head['pca_components'].T
    logit = z @ np.asarray(head['coefficient']).reshape(-1) + float(np.asarray(head['intercept']).reshape(-1)[0])
    return 1 / (1 + np.exp(-np.clip(logit, -500, 500)))

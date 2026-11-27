"""GPU training and batched scoring for research_v2 (BiGRU autoencoder and STG-NF flow).

GPU-aware by design: training windows live on the device as one tensor (fp16 on
CUDA), batches are drawn by index on the device, the autoencoder trains under
``torch.autocast`` with a ``GradScaler`` (bf16 on Ampere+ without a scaler), and
inference runs in large batches under ``torch.inference_mode``. All of it also runs
on CPU (slowly) so the unit tests work without a GPU.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from src.models.bigru_autoencoder import BiGRUAutoencoder
from src.models.stg_nf import STGNF, window_normalise

FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
FLIP_INDEX = list(range(17))
for _a, _b in FLIP_PAIRS:
    FLIP_INDEX[_a], FLIP_INDEX[_b] = _b, _a


def device_and_dtype():
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        amp = torch.bfloat16 if torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        torch.backends.cudnn.benchmark = True
        return dev, amp
    return torch.device('cpu'), None


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(windows, idx, device, fp32=False):
    """(xy, vis) tensors for a subset of a WindowSet; fp16 storage on CUDA for training data."""
    store = torch.float16 if device.type == 'cuda' and not fp32 else torch.float32
    xy = torch.from_numpy(np.ascontiguousarray(windows.xy[idx], dtype=np.float32)).to(device, store)
    vis = torch.from_numpy(np.ascontiguousarray(windows.vis[idx])).to(device)
    return xy, vis


def augment(xy, vis, cfg, generator):
    """Light pose augmentation on device. Returns (input xy, target xy, input vis, target vis).

    flip (with left/right joint swap), scale, rotation and jitter apply to both input and
    target; joint dropout removes joints from the *input only* (denoising), the loss still
    covers every originally visible joint."""
    b = xy.shape[0]
    dev = xy.device
    xy = xy.float()
    if cfg.get('flip', 0) > 0:
        flip = torch.rand(b, device=dev, generator=generator) < cfg['flip']
        flipped = xy[:, :, FLIP_INDEX].clone()
        flipped[..., 0] = -flipped[..., 0]
        xy = torch.where(flip[:, None, None, None], flipped, xy)
        vis = torch.where(flip[:, None, None], vis[:, :, FLIP_INDEX], vis)
    scale = 1 + (torch.rand(b, device=dev, generator=generator) * 2 - 1) * cfg.get('scale', 0)
    angle = (torch.rand(b, device=dev, generator=generator) * 2 - 1) * math.radians(cfg.get('rotate_deg', 0))
    cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale
    x, y = xy[..., 0], xy[..., 1]
    xy = torch.stack([cos[:, None, None] * x - sin[:, None, None] * y,
                      sin[:, None, None] * x + cos[:, None, None] * y], -1)
    if cfg.get('jitter', 0) > 0:
        xy = xy + torch.randn(xy.shape, device=dev, generator=generator) * cfg['jitter']
    xy = xy * vis.unsqueeze(-1)
    vis_in = vis
    if cfg.get('joint_dropout', 0) > 0:
        keep = torch.rand(vis.shape, device=dev, generator=generator) >= cfg['joint_dropout']
        vis_in = vis & keep
    return xy * vis_in.unsqueeze(-1), xy, vis_in, vis


def _flat(xy, vis):
    b, t = xy.shape[:2]
    return xy.reshape(b, t, 34).float(), vis.unsqueeze(-1).expand(-1, -1, -1, 2).reshape(b, t, 34).float()


def masked_mse(recon, target, mask):
    den = mask.sum(dim=(1, 2)).clamp_min(1)
    return (((recon.float() - target) ** 2) * mask).sum(dim=(1, 2)) / den


def make_ae(cfg):
    return BiGRUAutoencoder(34, cfg['hidden_size'], cfg['num_layers'], cfg['dropout'])


@torch.no_grad()
def _eval_loss(model, kind, xy, vis, batch, amp):
    model.eval()
    total, n = 0.0, 0
    for s in range(0, len(xy), batch):
        x, v = xy[s:s + batch], vis[s:s + batch]
        if kind == 'ae':
            xf, mf = _flat(x, v)
            with torch.autocast(xy.device.type, dtype=amp, enabled=amp is not None):
                recon = model(xf)
            loss = masked_mse(recon, xf, mf)
        else:
            loss = model.nll(window_normalise(x.float(), v))
        total += float(loss.sum())
        n += len(x)
    return total / max(n, 1)


def train_model(kind, train_xy, train_vis, val_xy, val_vis, cfg, seed, log=print):
    """kind: 'ae' (BiGRU autoencoder) or 'flow' (STG-NF). Returns (model, history, best_epoch).

    Checkpoint selection = lowest *normal-validation* loss; test data never enters."""
    seed_all(seed)
    dev = train_xy.device
    amp = cfg.get('amp_dtype')
    model = (make_ae(cfg) if kind == 'ae' else STGNF(2, cfg.get('flow_hidden', 32), cfg.get('flow_steps', 8))).to(dev)
    batch, epochs = int(cfg['batch_size']), int(cfg['epochs'])
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg['learning_rate']), weight_decay=float(cfg.get('weight_decay', 0)))
    steps_per_epoch = max(1, len(train_xy) // batch)
    sched = None
    if cfg.get('cosine', False):
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(cfg['learning_rate']), total_steps=epochs * steps_per_epoch,
                                                    pct_start=0.05, anneal_strategy='cos')
    use_scaler = kind == 'ae' and amp == torch.float16
    scaler = torch.amp.GradScaler(dev.type, enabled=use_scaler)
    gen = torch.Generator(device=dev).manual_seed(seed)
    aug = cfg.get('augment') or {}
    best, best_state, best_epoch, history, patience = float('inf'), None, 0, [], 0
    started = time.perf_counter()
    if kind == 'flow':                       # data-dependent ActNorm init on one large batch
        with torch.no_grad():
            init = torch.randperm(len(train_xy), device=dev, generator=gen)[:min(len(train_xy), 4096)]
            model.train()
            model.nll(window_normalise(train_xy[init].float(), train_vis[init]))
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(train_xy), device=dev, generator=gen)
        total, count = 0.0, 0
        for s in range(0, steps_per_epoch * batch, batch):
            idx = perm[s:s + batch]
            x, v = train_xy[idx], train_vis[idx]
            opt.zero_grad(set_to_none=True)
            if kind == 'ae':
                if aug:
                    x_in, x_t, v_in, v_t = augment(x, v, aug, gen)
                else:
                    x_in = x_t = x.float(); v_in = v_t = v
                xf_in, _ = _flat(x_in, v_in)
                xf_t, mf_t = _flat(x_t, v_t)
                with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
                    recon = model(xf_in)
                loss = masked_mse(recon, xf_t, mf_t).mean()
            else:
                if aug:
                    _, x, _, v = augment(x, v, {k: aug[k] for k in aug if k != 'joint_dropout'}, gen)
                loss = model.nll(window_normalise(x.float(), v)).mean()
            if not torch.isfinite(loss):
                raise ValueError(f'Non-finite training loss at epoch {epoch}')
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
            total += float(loss.detach()) * len(idx)
            count += len(idx)
        val = _eval_loss(model, kind, val_xy, val_vis, int(cfg.get('eval_batch_size', 4096)), amp)
        history.append(dict(epoch=epoch, train_loss=total / max(count, 1), validation_loss=val,
                            elapsed_seconds=time.perf_counter() - started))
        log(f'  epoch {epoch}/{epochs} train={total / max(count, 1):.5f} val={val:.5f} ({time.perf_counter() - started:.0f}s)')
        if val < best - 1e-7:
            best, best_epoch, patience = val, epoch, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            patience += 1
            if cfg.get('patience') and patience >= int(cfg['patience']):
                log(f'  early stop at epoch {epoch} (best {best_epoch})')
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


@torch.inference_mode()
def ae_errors(model, xy, vis, batch=4096, amp=None):
    """Per-window, per-frame, per-joint squared error (N, T, 17) as float16 numpy + support mask."""
    model.eval()
    errs = []
    for s in range(0, len(xy), batch):
        x, v = xy[s:s + batch], vis[s:s + batch]
        xf, _ = _flat(x, v)
        with torch.autocast(xy.device.type, dtype=amp, enabled=amp is not None):
            recon = model(xf)
        e = (recon.float() - xf) ** 2
        e = e.reshape(len(x), x.shape[1], 17, 2).mean(-1) * v
        errs.append(e.half().cpu().numpy())
    err = np.concatenate(errs) if errs else np.zeros((0, xy.shape[1], 17), np.float16)
    if not np.isfinite(err.astype(np.float32)).all():
        raise ValueError('Non-finite reconstruction errors')
    return err


@torch.inference_mode()
def flow_scores(model, xy, vis, batch=4096):
    model.eval()
    out = []
    for s in range(0, len(xy), batch):
        out.append(model.nll(window_normalise(xy[s:s + batch].float(), vis[s:s + batch])).cpu().numpy())
    scores = np.concatenate(out).astype(np.float64)
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite flow scores')
    return scores

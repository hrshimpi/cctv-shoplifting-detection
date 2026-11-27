"""STG-NF-style normalizing flow on the COCO-17 spatio-temporal pose graph (research_v2 experiment F).

After Hirschorn & Avidan, "Normalizing Flows for Human Pose Anomaly Detection"
(ICCV 2023). Input is one window of 2D poses as (B, 2, T, 17). Each flow step is
ActNorm -> invertible 1x1 channel mix -> affine coupling whose scale/shift net is
made of spatio-temporal graph convolutions (graph conv over the skeleton, then a
temporal conv). The base density is a standard normal, and the anomaly score is
the negative log-likelihood per dimension. Log-determinants are kept in float32.
"""
from __future__ import annotations

import math

import torch
from torch import nn

COCO_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
              (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def coco_adjacency(num_joints: int = 17) -> torch.Tensor:
    a = torch.eye(num_joints)
    for i, j in COCO_EDGES:
        a[i, j] = a[j, i] = 1.0
    d = a.sum(1).rsqrt()
    return d[:, None] * a * d[None]


class STGCN(nn.Module):
    def __init__(self, cin, cout, kernel_t=9):
        super().__init__()
        self.register_buffer('A', coco_adjacency())
        self.spatial = nn.Conv2d(cin, cout, 1)
        self.temporal = nn.Conv2d(cout, cout, (kernel_t, 1), padding=(kernel_t // 2, 0))
        self.norm = nn.BatchNorm2d(cout)
        self.res = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):                                  # x: (B, C, T, V)
        h = torch.einsum('bctv,vw->bctw', self.spatial(x), self.A)
        return torch.relu(self.norm(self.temporal(torch.relu(h))) + self.res(x))


class ActNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                mean = x.mean(dim=(0, 2, 3), keepdim=True)
                std = x.std(dim=(0, 2, 3), keepdim=True)
                self.loc.data.copy_(-mean)
                self.logs.data.copy_(-torch.log(std + 1e-6))
                self.initialized.fill_(1)
        y = (x + self.loc) * torch.exp(self.logs)
        logdet = self.logs.sum() * x.shape[2] * x.shape[3]
        return y, logdet.expand(x.shape[0])


class InvConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        w = torch.linalg.qr(torch.randn(channels, channels))[0]
        self.weight = nn.Parameter(w)

    def forward(self, x):
        y = torch.einsum('oc,bctv->botv', self.weight, x)
        logdet = torch.slogdet(self.weight.float())[1] * x.shape[2] * x.shape[3]
        return y, logdet.expand(x.shape[0])


class Coupling(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.c1 = channels // 2
        c2 = channels - self.c1
        self.net = nn.Sequential(STGCN(self.c1, hidden), STGCN(hidden, hidden))
        self.out = nn.Conv2d(hidden, 2 * c2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        xa, xb = x[:, :self.c1], x[:, self.c1:]
        h = self.out(self.net(xa))
        log_s, t = h.chunk(2, dim=1)
        log_s = 2.0 * torch.tanh(log_s / 2.0)               # clamp scales for stability
        yb = xb * torch.exp(log_s) + t
        return torch.cat([xa, yb], 1), log_s.flatten(1).sum(1)


class STGNF(nn.Module):
    def __init__(self, channels: int = 2, hidden: int = 32, steps: int = 8):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(steps):
            self.layers.extend([ActNorm(channels), InvConv(channels), Coupling(channels, hidden)])

    def forward(self, x):
        """x: (B, 2, T, V) -> (z, log-det per sample)."""
        logdet = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
        for layer in self.layers:
            x, ld = layer(x)
            logdet = logdet + ld.float()
        return x, logdet

    def nll(self, x):
        """Negative log-likelihood per dimension, (B,)."""
        z, logdet = self(x.float())
        dims = z[0].numel()
        log_pz = -0.5 * (z.float() ** 2).flatten(1).sum(1) - 0.5 * dims * math.log(2 * math.pi)
        return -(log_pz + logdet) / dims


def window_normalise(xy: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    """(B, T, 17, 2) + (B, T, 17) -> (B, 2, T, 17): centre on the window's visible-joint mean,
    scale by the std of visible y (STG-NF per-segment normalisation); missing joints -> 0."""
    m = vis.unsqueeze(-1).float()
    count = m.sum(dim=(1, 2)).clamp_min(1)
    centre = (xy * m).sum(dim=(1, 2)) / count
    centred = (xy - centre[:, None, None]) * m
    var_y = (centred[..., 1] ** 2 * vis.float()).sum(dim=(1, 2)) / count[:, 0]
    scale = var_y.clamp_min(1e-4).sqrt()
    return (centred / scale[:, None, None, None]).permute(0, 3, 1, 2).contiguous()

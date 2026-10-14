"""Batched GPU augmentation on the [wafer, fail] base tensor (spec 5.4).

Allowed: D4 symmetries (exact), small rotation (nearest), die-flip noise. Forbidden: crops,
translations, scaling, colour jitter -- position is part of the label.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def apply_d4(x: torch.Tensor, t: int) -> torch.Tensor:
    """Transform t in 0..7: rotate by 90*(t % 4), then flip left-right if t >= 4."""
    y = torch.rot90(x, t % 4, dims=(-2, -1))
    return y.flip(-1) if t >= 4 else y


def invert_d4(x: torch.Tensor, t: int) -> torch.Tensor:
    y = x.flip(-1) if t >= 4 else x
    return torch.rot90(y, -(t % 4), dims=(-2, -1))


def random_d4(x: torch.Tensor) -> torch.Tensor:
    t = torch.randint(0, 8, (x.shape[0],), device=x.device)
    out = torch.empty_like(x)
    for k in range(8):
        m = t == k
        if m.any():
            out[m] = apply_d4(x[m], k)
    return out


def small_rotation(x: torch.Tensor, max_deg: float = 15.0, p: float = 0.5) -> torch.Tensor:
    """Rotate a random subset by U(-max_deg, max_deg) with nearest sampling. Outside the original
    grid becomes 0 in both channels (no die), so no dies appear outside the rotated wafer."""
    b = x.shape[0]
    sel = torch.rand(b, device=x.device) < p
    if not sel.any():
        return x
    xs = x[sel]
    ang = (torch.rand(xs.shape[0], device=x.device) * 2 - 1) * math.radians(max_deg)
    cos, sin = torch.cos(ang), torch.sin(ang)
    zeros = torch.zeros_like(cos)
    theta = torch.stack([torch.stack([cos, -sin, zeros], 1), torch.stack([sin, cos, zeros], 1)], 1)
    grid = F.affine_grid(theta, list(xs.shape), align_corners=False)
    rot = F.grid_sample(xs, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    rot[:, 1] = torch.minimum(rot[:, 1], rot[:, 0])
    out = x.clone()
    out[sel] = rot
    return out


def die_flip(x: torch.Tensor, max_rate: float) -> torch.Tensor:
    """At a fraction U(0, max_rate) of in-wafer pixels, swap the pass/fail mix:
    fail <- wafer - fail (a binary pass<->fail flip for encoding A)."""
    if max_rate <= 0:
        return x
    b = x.shape[0]
    rate = torch.rand(b, 1, 1, device=x.device) * max_rate
    wafer, fail = x[:, 0], x[:, 1]
    flip = (torch.rand_like(wafer) < rate) & (wafer > 0)
    return torch.stack([wafer, torch.where(flip, wafer - fail, fail)], dim=1)


def augment(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    """cfg = {"d4": bool, "small_rot": bool, "die_flip": float}"""
    if cfg.get("d4", True):
        x = random_d4(x)
    if cfg.get("small_rot", False):
        x = small_rotation(x)
    if cfg.get("die_flip", 0.0):
        x = die_flip(x, float(cfg["die_flip"]))
    return x

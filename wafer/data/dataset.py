"""GPU-resident wafer data (spec 4.2, 5.2). No DataLoader: the whole npz lives on the device as
uint8, and each step indexes a batch, converts it to float and builds the input channels.

Augmentation always works on a 2-channel *base* tensor [wafer, fail] (fractions for encoding B,
0/1 for encoding A); `to_input` then derives the model's channels, so every encoding gets the
same augmentation code and pass = wafer - fail stays consistent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

ENCODINGS = ("area", "nearest", "area_r")


def base_from_area_u8(x_u8: torch.Tensor) -> torch.Tensor:
    """(B,2,S,S) uint8 x255 -> (B,2,S,S) float [wafer_frac, fail_frac]."""
    x = x_u8.float() / 255.0
    return torch.stack([x[:, 0], torch.minimum(x[:, 1], x[:, 0])], dim=1)


def base_from_nearest(x: torch.Tensor) -> torch.Tensor:
    """(B,S,S) {0,1,2} -> (B,2,S,S) float [wafer, fail] in {0,1}."""
    return torch.stack([(x > 0).float(), (x == 2).float()], dim=1)


def radius_channel(wafer: torch.Tensor) -> torch.Tensor:
    """(B,S,S) wafer fraction -> (B,S,S) distance from the wafer centroid / equivalent-area radius,
    clipped to [0, 1.2]. Invariant under D4 (rotating the wafer rotates the channel)."""
    b, s, _ = wafer.shape
    coords = torch.arange(s, device=wafer.device, dtype=wafer.dtype) + 0.5
    mass = wafer.sum(dim=(1, 2)).clamp_min(1e-6)
    cy = (wafer.sum(dim=2) * coords).sum(dim=1) / mass
    cx = (wafer.sum(dim=1) * coords).sum(dim=1) / mass
    radius = torch.sqrt(mass / torch.pi).clamp_min(1e-3)
    d = torch.sqrt((coords.view(1, s, 1) - cy.view(b, 1, 1)) ** 2 + (coords.view(1, 1, s) - cx.view(b, 1, 1)) ** 2)
    return (d / radius.view(b, 1, 1)).clamp(0.0, 1.2)


def to_input(base: torch.Tensor, encoding: str) -> torch.Tensor:
    """(B,2,S,S) [wafer, fail] -> model input."""
    wafer, fail = base[:, 0], base[:, 1]
    passf = (wafer - fail).clamp_min(0.0)
    if encoding == "area":
        return torch.stack([wafer, passf, fail], dim=1)
    if encoding == "nearest":
        return torch.stack([1.0 - wafer, passf, fail], dim=1)
    if encoding == "area_r":
        return torch.stack([wafer, passf, fail, radius_channel(wafer)], dim=1)
    raise ValueError(f"unknown encoding {encoding!r}; options {ENCODINGS}")


def aux_features(n_die, orig_h, orig_w, n_fail) -> np.ndarray:
    """E8 scalars, all computable from the raw map at serving time: log die count (~unit scale),
    log aspect ratio, fail fraction."""
    n_die = np.maximum(np.asarray(n_die, dtype=np.float32), 1)
    return np.stack([np.log(n_die) / 8.0,
                     np.log(np.asarray(orig_h, np.float32) / np.asarray(orig_w, np.float32)),
                     np.asarray(n_fail, np.float32) / n_die], axis=1).astype(np.float32)


class WaferData:
    """All labelled wafers of one npz, on `device`. Rows align with metadata.parquet."""

    def __init__(self, npz_path: str | Path, meta_path: str | Path, encoding: str = "area",
                 device: str | torch.device = "cpu", aux: bool = False):
        if encoding not in ENCODINGS:
            raise ValueError(f"unknown encoding {encoding!r}; options {ENCODINGS}")
        z = np.load(npz_path)
        meta = pd.read_parquet(meta_path)
        if not np.array_equal(z["uid"], meta["uid"].to_numpy()):
            raise ValueError(f"{npz_path} rows do not align with {meta_path} uids")
        self.encoding, self.device, self.meta = encoding, torch.device(device), meta
        key = "X_nearest" if encoding == "nearest" else "X_area"
        self.x = torch.from_numpy(np.ascontiguousarray(z[key])).to(self.device)
        self.y = torch.from_numpy(z["y"].astype(np.int64)).to(self.device)
        self.uid = z["uid"].astype(np.int64)
        self.y_np = z["y"].astype(np.int64)
        self.size = int(self.x.shape[-1])
        self.aux = None
        if aux:
            a = aux_features(meta.n_die, meta.orig_h, meta.orig_w, meta.n_fail)
            self.aux = torch.from_numpy(a).to(self.device)

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.meta["split"].to_numpy() == split)

    def base(self, idx) -> torch.Tensor:
        idx = torch.as_tensor(idx, device=self.device)
        x = self.x[idx]
        return base_from_nearest(x) if self.encoding == "nearest" else base_from_area_u8(x)

    def batch(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """(base [B,2,S,S], y [B], aux [B,3] | None) for row indices `idx`."""
        idx_t = torch.as_tensor(idx, device=self.device)
        return self.base(idx_t), self.y[idx_t], (self.aux[idx_t] if self.aux is not None else None)

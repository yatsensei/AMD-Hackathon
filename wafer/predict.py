"""Batched forward passes shared by training, evaluation, export and serving."""
from __future__ import annotations

import numpy as np
import torch

from wafer.data.augment import apply_d4
from wafer.data.dataset import to_input


@torch.no_grad()
def forward_base(model, base: torch.Tensor, encoding: str, aux: torch.Tensor | None = None,
                 tta: bool = False, amp: bool = False) -> torch.Tensor:
    """Logits (no TTA) or log of D4-averaged probabilities (TTA) for a base batch [B,2,S,S]."""
    dev = base.device.type
    ctx = torch.autocast(dev, dtype=torch.bfloat16) if amp and dev == "cuda" else torch.autocast(dev, enabled=False)
    with ctx:
        if not tta:
            return model(to_input(base, encoding), aux).float()
        probs = torch.stack([torch.softmax(model(to_input(apply_d4(base, t), encoding), aux).float(), 1)
                             for t in range(8)]).mean(0)
    return probs.clamp_min(1e-12).log()


@torch.no_grad()
def predict_rows(model, data, idx: np.ndarray, tta: bool = False, amp: bool = False,
                 batch_size: int = 4096) -> np.ndarray:
    """Outputs for rows `idx` of a WaferData, as float32 numpy (N, 9)."""
    model.eval()
    out = []
    for i in range(0, len(idx), batch_size):
        base, _, aux = data.batch(idx[i:i + batch_size])
        out.append(forward_base(model, base, data.encoding, aux, tta=tta, amp=amp).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 9), np.float32)

"""Loss options (spec 5.6): ce, weighted_ce, focal (gamma 2, class weights as alpha), ce_ls (0.05)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits.float(), dim=1).gather(1, y[:, None])[:, 0]
        loss = -((1 - logp.exp()) ** self.gamma) * logp
        if self.alpha is None:
            return loss.mean()
        a = self.alpha[y]
        return (a * loss).sum() / a.sum()


def make_loss(name: str, weights: torch.Tensor) -> nn.Module:
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weights)
    if name == "ce_ls":
        return nn.CrossEntropyLoss(label_smoothing=0.05)
    if name == "focal":
        return FocalLoss(alpha=weights, gamma=2.0)
    raise ValueError(f"unknown loss {name!r}; options: ce, weighted_ce, focal, ce_ls")

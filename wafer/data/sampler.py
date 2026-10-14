"""Per-epoch `none` downsampling, class weights and the effective training prior (spec 5.3).

The effective prior is what the model's logits implicitly assume: per-epoch sample counts times
loss weights. Calibration subtracts it again (spec 7.4 step 2).
"""
from __future__ import annotations

import numpy as np

from wafer.constants import CLASSES, NONE_IDX

N_CLASSES = len(CLASSES)
WEIGHTED_LOSSES = {"weighted_ce", "focal"}


def class_counts(y: np.ndarray) -> np.ndarray:
    return np.bincount(np.asarray(y), minlength=N_CLASSES).astype(np.float64)


def class_weights(train_counts: np.ndarray, loss: str) -> np.ndarray:
    """w_c = (N / n_c) ** 0.5 normalised to mean 1 for weighted losses, else ones."""
    if loss not in WEIGHTED_LOSSES:
        return np.ones(N_CLASSES)
    n = np.maximum(train_counts, 1.0)
    w = np.sqrt(n.sum() / n)
    return w / w.mean()


def epoch_counts(train_counts: np.ndarray, sampler: str, none_per_epoch: int) -> np.ndarray:
    c = train_counts.copy()
    if sampler == "none_downsample":
        c[NONE_IDX] = min(c[NONE_IDX], none_per_epoch)
    elif sampler != "none":
        raise ValueError(f"unknown sampler {sampler!r}; options: none_downsample, none")
    return c


def effective_prior(train_counts: np.ndarray, sampler: str, none_per_epoch: int, loss: str) -> np.ndarray:
    p = epoch_counts(train_counts, sampler, none_per_epoch) * class_weights(train_counts, loss)
    return p / p.sum()


def epoch_indices(train_idx: np.ndarray, y_train: np.ndarray, sampler: str, none_per_epoch: int,
                  rng: np.random.Generator) -> np.ndarray:
    """All defect wafers + a fresh random sample of `none_per_epoch` none wafers, shuffled."""
    train_idx = np.asarray(train_idx)
    if sampler == "none":
        out = train_idx.copy()
    else:
        is_none = np.asarray(y_train) == NONE_IDX
        none_idx = train_idx[is_none]
        k = min(len(none_idx), none_per_epoch)
        out = np.concatenate([train_idx[~is_none], rng.choice(none_idx, size=k, replace=False)])
    rng.shuffle(out)
    return out

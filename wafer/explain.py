"""Grad-CAM attribution and the defect box (spec Section 8). Attribution, not detection:
the box marks the fail dies that drove the decision."""
from __future__ import annotations

import cv2
import numpy as np

from wafer.constants import FAIL

WHOLE_WAFER = {"Random", "Near-full", "Edge-Ring"}     # patterns that span the wafer


def grad_cam(model, x, class_idx: int) -> np.ndarray:
    """Grad-CAM on `model.cam_layer` for one input x (1, C, S, S). Returns (S, S) float32 in [0,1]."""
    import torch
    import torch.nn.functional as F

    store = {}

    def hook(_module, _inp, out):
        out.retain_grad()
        store["act"] = out

    handle = model.cam_layer.register_forward_hook(hook)
    try:
        was_training = model.training
        model.eval()
        with torch.enable_grad():
            x = x.detach().requires_grad_(False)
            logits = model(x)
            model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()
        model.train(was_training)
    finally:
        handle.remove()
    act, grad = store["act"], store["act"].grad
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().float().cpu().numpy()
    peak = cam.max()
    return (cam / peak).astype(np.float32) if peak > 0 else np.zeros_like(cam, dtype=np.float32)


def to_original(heatmap: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize an S x S heatmap to the original wafer's H x W, clipped to [0,1]."""
    h, w = shape
    out = cv2.resize(np.asarray(heatmap, dtype=np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(out, 0.0, 1.0)


def wafer_bbox(map2d: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(np.asarray(map2d) > 0)
    if len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def defect_bbox(map2d: np.ndarray, heatmap: np.ndarray, pattern: str, thresh: float = 0.25) -> list[int] | None:
    """Box around the fail-die cluster that drove the decision, in original coordinates (inclusive).

    1. Keep fail dies where the Grad-CAM heat is >= `thresh` x its peak (drops background noise
       far from the evidence).
    2. Group them into clusters (3x3 closing, so a dotted scratch or a sparse arc is one cluster).
    3. Return the box of the cluster with the most heat, in full.
    Low-heat dies far away never join, and a long arc or blob is boxed whole rather than only its
    hottest part. `none` -> None; Random / Near-full / Edge-Ring -> the wafer's own box.
    """
    m = np.asarray(map2d)
    if pattern == "none":
        return None
    if pattern in WHOLE_WAFER:
        return wafer_bbox(m)
    fail = (m == FAIL)
    if not fail.any():
        return None
    h = np.asarray(heatmap, dtype=np.float32)
    peak = float(h.max()) if h.size else 0.0
    keep = (fail & (h >= thresh * peak)) if peak > 0 else fail
    if not keep.any():
        keep = fail
    clusters = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) & (m > 0)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(clusters.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    score = np.bincount(labels.ravel(), weights=(h * keep).ravel(), minlength=n)[1:]
    if score.max() <= 0:
        score = stats[1:, cv2.CC_STAT_AREA].astype(float)
    k = 1 + int(np.argmax(score))
    x, y, bw, bh = (int(v) for v in stats[k, :4])
    return [x, y, x + bw - 1, y + bh - 1]

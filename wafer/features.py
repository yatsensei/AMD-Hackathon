"""Hand-crafted wafer features for the E0b classical baseline (spec 6.1).

Computed from the same 64x64 area-fraction encoding the CNN sees, and rotation/flip invariant
(sector densities are sorted), so the comparison with the CNN is about the model, not the input.
Features: overall fail fraction; fail density in 5 radial bands; sorted fail density in 8 angular
sectors of the outer ring and of the inner disc; centre/edge ratio; and shape statistics of the
fail-die blobs (count, largest blob share, its radius, elongation and fill).
"""
from __future__ import annotations

import cv2
import numpy as np

BANDS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.3)]
N_SECTORS = 8


def _radius_angle(wafer: np.ndarray):
    s = wafer.shape[0]
    c = np.arange(s) + 0.5
    mass = max(wafer.sum(), 1e-6)
    cy, cx = (wafer.sum(1) * c).sum() / mass, (wafer.sum(0) * c).sum() / mass
    yy, xx = np.meshgrid(c, c, indexing="ij")
    r = np.hypot(yy - cy, xx - cx) / max(np.sqrt(mass / np.pi), 1e-3)
    return r, np.arctan2(yy - cy, xx - cx)


def _density(fail, wafer, mask):
    w = (wafer * mask).sum()
    return float((fail * mask).sum() / w) if w > 0 else 0.0


def wafer_features(x_area_u8: np.ndarray) -> np.ndarray:
    """(2, S, S) uint8 [wafer, fail] x255 -> 1D float32 feature vector."""
    wafer = x_area_u8[0].astype(np.float32) / 255
    fail = np.minimum(x_area_u8[1].astype(np.float32) / 255, wafer)
    r, theta = _radius_angle(wafer)
    f = [_density(fail, wafer, np.ones_like(wafer))]
    f += [_density(fail, wafer, (r >= a) & (r < b)) for a, b in BANDS]
    edges = np.linspace(-np.pi, np.pi, N_SECTORS + 1)
    for zone in (r >= 0.7, r < 0.7):
        sec = np.sort([_density(fail, wafer, zone & (theta >= edges[i]) & (theta < edges[i + 1]))
                       for i in range(N_SECTORS)])
        f += list(sec) + [float(sec.std())]
    centre, edge = _density(fail, wafer, r < 0.35), _density(fail, wafer, r >= 0.8)
    f.append(centre / (edge + 1e-3))

    binary = (fail >= 0.25).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    total = max(int(binary.sum()), 1)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[k, cv2.CC_STAT_AREA])
        ys, xs = np.nonzero(labels == k)
        cov = np.cov(np.stack([xs, ys])) if area > 2 else np.eye(2)
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        elong = float(np.sqrt(ev[0] / max(ev[1], 1e-3)))
        bw, bh = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        fill = area / max(bw * bh, 1)
        cy, cx = int(round(cents[k][1])), int(round(cents[k][0]))
        blob_r = float(r[min(cy, r.shape[0] - 1), min(cx, r.shape[1] - 1)])
        f += [n - 1, area / total, area / max(wafer.sum(), 1), elong, fill, blob_r]
    else:
        f += [0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return np.asarray(f, dtype=np.float32)


def all_features(x_area_u8: np.ndarray) -> np.ndarray:
    return np.stack([wafer_features(x) for x in x_area_u8])

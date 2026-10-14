"""Shared wafer rendering (spec 3.9). YOLO PNGs, the dashboard and pitch figures all use these,
so a wafer looks the same everywhere. All functions return H x W x 3 uint8 RGB."""
from __future__ import annotations

import cv2
import numpy as np

PALETTE = {0: (20, 20, 20), 1: (60, 160, 90), 2: (220, 60, 60)}  # no-die, pass, fail
_LUT = np.array([PALETTE[0], PALETTE[1], PALETTE[2]], dtype=np.uint8)


def render_wafer(map2d: np.ndarray, size: int | None = None) -> np.ndarray:
    """Raw {0,1,2} map -> RGB image; optionally resized (nearest) to size x size."""
    a = np.asarray(map2d)
    if a.ndim != 2 or a.size == 0:
        raise ValueError(f"map2d must be a non-empty 2D array, got shape {a.shape}")
    if a.min() < 0 or a.max() > 2:
        raise ValueError("map2d values must be in {0, 1, 2}")
    img = _LUT[a.astype(np.intp)]
    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    return img


def render_fractions(wafer_frac: np.ndarray, fail_frac: np.ndarray) -> np.ndarray:
    """Encoding-B fractions -> RGB, blending the palette by the share of no-die/pass/fail dies.

    A pixel that is 30% fail dies shows 30% of the fail colour, so thin scratches stay visible
    at low resolution (unlike render_wafer on a nearest-resized map).
    """
    w = np.clip(np.asarray(wafer_frac, dtype=np.float32), 0, 1)
    f = np.minimum(np.clip(np.asarray(fail_frac, dtype=np.float32), 0, 1), w)
    weights = np.stack([1 - w, w - f, f], axis=-1)                     # H x W x 3 (no-die, pass, fail)
    img = weights @ _LUT.astype(np.float32)                            # H x W x RGB
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def render_heatmap(map2d: np.ndarray, heatmap: np.ndarray, bbox=None, alpha: float = 0.5) -> np.ndarray:
    """Overlay a [0,1] heatmap (same H x W as map2d) on the wafer, inside the wafer only.

    bbox = [x0, y0, x1, y1] in map coordinates (inclusive), drawn in white if given.
    """
    base = render_wafer(map2d)
    h = np.asarray(heatmap, dtype=np.float32)
    if h.shape != base.shape[:2]:
        raise ValueError(f"heatmap shape {h.shape} != map shape {base.shape[:2]}")
    colour = cv2.applyColorMap(np.rint(np.clip(h, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
    inside = (np.asarray(map2d) > 0)[..., None]
    out = np.where(inside, (1 - alpha) * base + alpha * colour, base)
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    if bbox is not None:
        x0, y0, x1, y1 = (int(v) for v in bbox)
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 1)
    return out

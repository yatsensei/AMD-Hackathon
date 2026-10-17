"""Exact YOLO-cls preprocessing from the teammate train/infer pipeline.

Must stay bit-compatible with how `yolo_cls` PNGs were written:
nearest resize to `size`, then BGR colouring of {0,1,2} die states.
"""
from __future__ import annotations

import cv2
import numpy as np

# Colours for the rendered image (BGR): no chip / pass / fail — unchanged from training.
COLORS_BGR = {0: (0, 0, 0), 1: (90, 140, 60), 2: (40, 220, 255)}
DEFAULT_SIZE = 128


def render(wmap: np.ndarray, size: int = DEFAULT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Return (nearest-resized label map, BGR uint8 image) for Ultralytics classify."""
    wmap = np.asarray(wmap, dtype=np.uint8)
    small = cv2.resize(wmap, (size, size), interpolation=cv2.INTER_NEAREST)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for v, c in COLORS_BGR.items():
        img[small == v] = c
    return small, img

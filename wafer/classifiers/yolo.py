"""YOLO backend stub. Real adapter lands after YOLO code and weights are available."""
from __future__ import annotations

import numpy as np

from wafer.classifiers.base import ClassifierNotAvailableError

_MSG = "YOLO is not available yet"


class YoloClassifier:
    """Registered under name 'yolo' so selection is real; every call fails clearly."""

    name = "yolo"

    def __init__(self):
        raise ClassifierNotAvailableError(_MSG)

    def classify(self, wafer_map: np.ndarray) -> dict:  # pragma: no cover
        raise ClassifierNotAvailableError(_MSG)

    def classify_batch(self, maps: list[np.ndarray], batch_size: int = 512) -> list[dict]:  # pragma: no cover
        raise ClassifierNotAvailableError(_MSG)

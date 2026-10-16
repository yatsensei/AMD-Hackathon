"""CNN backend: wraps wafer.infer, preserving calibration and needs_review behaviour."""
from __future__ import annotations

from typing import Callable

import numpy as np


class CnnClassifier:
    """Adapter around wafer.infer (or an injectable classify pair for tests)."""

    name = "cnn"

    def __init__(self, classify_fn: Callable | None = None,
                 classify_batch_fn: Callable | None = None):
        self._classify = classify_fn
        self._classify_batch = classify_batch_fn

    def _fns(self):
        if self._classify is None or self._classify_batch is None:
            from wafer.infer import classify, classify_batch
            self._classify = self._classify or classify
            self._classify_batch = self._classify_batch or classify_batch
        return self._classify, self._classify_batch

    def _tag(self, result: dict) -> dict:
        out = dict(result)
        out["classifier"] = self.name
        # Keep CNN-specific keys top-level for back-compat; also mirror under extras.
        extras = {k: out[k] for k in ("top2", "defect_prob") if k in out}
        out["extras"] = {**extras, **out.get("extras", {})}
        return out

    def classify(self, wafer_map: np.ndarray) -> dict:
        classify_fn, _ = self._fns()
        return self._tag(classify_fn(wafer_map))

    def classify_batch(self, maps: list[np.ndarray], batch_size: int = 512) -> list[dict]:
        _, batch_fn = self._fns()
        try:
            results = batch_fn(maps, batch_size=batch_size)
        except TypeError:
            results = batch_fn(maps)
        return [self._tag(r) for r in results]

"""Common classifier backend contract for the agent and dashboard."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

COMMON_RESULT_KEYS = (
    "pattern",
    "confidence",
    "probs",
    "needs_review",
    "review_reasons",
    "model_version",
    "classifier",
)


class ClassifierNotAvailableError(RuntimeError):
    """Raised when a selected backend is registered but not implemented yet."""


@runtime_checkable
class ClassifierBackend(Protocol):
    """Pluggable wafer classifier. Agent and dashboard depend only on this surface."""

    name: str

    def classify(self, wafer_map: np.ndarray) -> dict:
        """Return a JSON-serialisable result with COMMON_RESULT_KEYS (plus optional extras)."""
        ...

    def classify_batch(self, maps: list[np.ndarray], batch_size: int = 512) -> list[dict]:
        ...

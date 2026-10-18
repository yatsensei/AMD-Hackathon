"""Agent session state: stream, replay clock, classifier backend and optional line context."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from wafer.classifiers.base import ClassifierBackend


@dataclass
class AgentContext:
    stream_df: pd.DataFrame
    as_of: str
    classifier: ClassifierBackend
    maps: dict[int, object] = field(default_factory=dict)  # optional uid -> map override (tests)
    last_history_full: dict[str, Any] | None = None  # full events kept locally, not sent to LLM
    # Engineer decisions (the dashboard passes its in-memory log); None = not available.
    decisions: Callable[[], list[dict]] | None = None
    # Grad-CAM explainer: map -> {"heatmap", "bbox", "pattern"}; None = wafer.infer.explain for CNN.
    explain_fn: Callable[[Any], dict] | None = None
    # Validation metrics.json of the line model; None = docs/results/<WAFER_MODEL_DIR name>_val.
    metrics: dict | None = None

    @property
    def classifier_name(self) -> str:
        return getattr(self.classifier, "name", "unknown")

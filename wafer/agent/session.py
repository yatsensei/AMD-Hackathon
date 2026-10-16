"""Agent session state: stream, replay clock, classifier backend."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from wafer.classifiers.base import ClassifierBackend


@dataclass
class AgentContext:
    stream_df: pd.DataFrame
    as_of: str
    classifier: ClassifierBackend
    maps: dict[int, object] = field(default_factory=dict)  # optional uid -> map override (tests)
    last_history_full: dict[str, Any] | None = None  # full events kept locally, not sent to LLM

    @property
    def classifier_name(self) -> str:
        return getattr(self.classifier, "name", "unknown")

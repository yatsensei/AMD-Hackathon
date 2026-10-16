"""Select a classifier backend by name. Default is CNN; YOLO never falls back to CNN."""
from __future__ import annotations

import os

from wafer.classifiers.base import ClassifierBackend, ClassifierNotAvailableError
from wafer.classifiers.cnn import CnnClassifier

DEFAULT_CLASSIFIER = "cnn"
KNOWN = ("cnn", "yolo")


def get_classifier(name: str | None = None, **kwargs) -> ClassifierBackend:
    """Return a backend. `name` overrides $WAFER_CLASSIFIER; unset defaults to cnn.

    Selecting 'yolo' raises ClassifierNotAvailableError — it does not substitute CNN.
    """
    selected = (name or os.environ.get("WAFER_CLASSIFIER") or DEFAULT_CLASSIFIER).strip().lower()
    if selected == "cnn":
        return CnnClassifier(**kwargs)
    if selected == "yolo":
        from wafer.classifiers.yolo import YoloClassifier
        return YoloClassifier()  # raises ClassifierNotAvailableError
    raise ValueError(f"unknown classifier {selected!r}; expected one of {KNOWN}")

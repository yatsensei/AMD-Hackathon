"""Classifier backend registry and CNN / YOLO stub behaviour."""
from __future__ import annotations

import numpy as np
import pytest

from wafer.classifiers import (
    COMMON_RESULT_KEYS,
    ClassifierNotAvailableError,
    CnnClassifier,
    get_classifier,
)
from wafer.infer_mock import classify, classify_batch
from tests.synthetic import draw_pattern


def test_default_is_cnn(monkeypatch):
    monkeypatch.delenv("WAFER_CLASSIFIER", raising=False)
    clf = get_classifier()
    assert clf.name == "cnn"


def test_env_selects_cnn(monkeypatch):
    monkeypatch.setenv("WAFER_CLASSIFIER", "cnn")
    assert get_classifier().name == "cnn"


def test_yolo_unavailable_no_cnn_fallback(monkeypatch):
    monkeypatch.setenv("WAFER_CLASSIFIER", "yolo")
    with pytest.raises(ClassifierNotAvailableError, match="YOLO is not available yet"):
        get_classifier()
    with pytest.raises(ClassifierNotAvailableError, match="YOLO is not available yet"):
        get_classifier("yolo")


def test_unknown_classifier():
    with pytest.raises(ValueError, match="unknown classifier"):
        get_classifier("resnet")


def test_cnn_adapter_common_fields():
    clf = CnnClassifier(classify_fn=classify, classify_batch_fn=classify_batch)
    m = draw_pattern("Edge-Ring", 40, 40, np.random.default_rng(0))
    r = clf.classify(m)
    assert r["classifier"] == "cnn"
    for k in COMMON_RESULT_KEYS:
        assert k in r
    assert isinstance(r["needs_review"], bool)
    assert isinstance(r["review_reasons"], list)
    assert "extras" in r
    batch = clf.classify_batch([m, m])
    assert len(batch) == 2 and all(b["classifier"] == "cnn" for b in batch)

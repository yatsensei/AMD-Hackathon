"""Classifier backend registry and CNN / YOLO behaviour."""
from __future__ import annotations

import numpy as np
import pytest

from wafer.classifiers import (
    COMMON_RESULT_KEYS,
    ClassifierNotAvailableError,
    CnnClassifier,
    get_classifier,
)
from wafer.constants import CLASSES
from wafer.infer_mock import classify, classify_batch
from tests.synthetic import draw_pattern


def test_default_is_cnn(monkeypatch):
    monkeypatch.delenv("WAFER_CLASSIFIER", raising=False)
    clf = get_classifier()
    assert clf.name == "cnn"


def test_env_selects_cnn(monkeypatch):
    monkeypatch.setenv("WAFER_CLASSIFIER", "cnn")
    assert get_classifier().name == "cnn"


def test_yolo_no_cnn_fallback_when_unavailable(monkeypatch, tmp_path):
    """Missing weights → ClassifierNotAvailableError; never returns a CNN instance."""
    monkeypatch.setenv("WAFER_CLASSIFIER", "yolo")
    with pytest.raises(ClassifierNotAvailableError, match="YOLO|ultralytics|missing"):
        get_classifier(weights_dir=tmp_path / "nope")
    with pytest.raises(ClassifierNotAvailableError):
        get_classifier("yolo", weights_dir=tmp_path / "nope")


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


def test_yolo_experimental_common_fields_if_available():
    """When weights + ultralytics are present, YOLO returns the common contract without CNN review rules."""
    ultralytics = pytest.importorskip("ultralytics")
    del ultralytics
    from pathlib import Path
    from wafer.env import REPO_ROOT
    weights = REPO_ROOT / "models" / "yolo-cls-exp" / "best.pt"
    if not weights.exists():
        pytest.skip("models/yolo-cls-exp/best.pt not present")
    clf = get_classifier("yolo")
    assert clf.name == "yolo"
    m = draw_pattern("Edge-Ring", 40, 40, np.random.default_rng(1))
    r = clf.classify(m)
    for k in COMMON_RESULT_KEYS:
        assert k in r
    assert r["classifier"] == "yolo"
    assert r["model_version"] == "yolo-cls-exp"
    assert r["pattern"] in CLASSES
    assert set(r["probs"]) == set(CLASSES)
    assert abs(sum(r["probs"].values()) - 1.0) < 1e-3
    assert r["needs_review"] is False
    assert r["review_reasons"] == []
    assert r["extras"]["calibrated"] is False
    assert r["extras"]["review_rules_validated"] is False
    assert r["extras"]["experimental"] is True
    batch = clf.classify_batch([m])
    assert len(batch) == 1 and batch[0]["classifier"] == "yolo"

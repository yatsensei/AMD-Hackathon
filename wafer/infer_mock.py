"""Stand-in for wafer.infer with the same signatures and output keys (spec 9.2).

Phases 5 and 6 build against this, then swap one import:
    from wafer.infer_mock import classify, classify_batch, explain    # today
    from wafer.infer import classify, classify_batch, explain         # once models/v0 exists

The "model" is a few hand-written rules on fail-die geometry, so outputs vary sensibly
between wafers (including needs_review cases). The numbers mean nothing: never quote them.
"""
from __future__ import annotations

import cv2
import numpy as np

from wafer.constants import CLASSES, FAIL
from wafer.data.preprocess import validate_map
from wafer.decision import result_dict
from wafer.explain import defect_bbox

MODEL_VERSION = "mock"
_THETA_NONE, _TAU = 0.5, 0.6


def _features(m: np.ndarray) -> dict:
    wafer = m > 0
    fail = (m == FAIL) & wafer
    ys, xs = np.nonzero(wafer)
    cy, cx = ys.mean(), xs.mean()
    ry, rx = max(np.ptp(ys) / 2, 1), max(np.ptp(xs) / 2, 1)
    yy, xx = np.mgrid[:m.shape[0], :m.shape[1]]
    r = np.hypot((yy - cy) / ry, (xx - cx) / rx)
    theta = np.arctan2(yy - cy, xx - cx)

    def frac(region):
        n = (wafer & region).sum()
        return float((fail & region).sum() / n) if n else 0.0

    edge = r > 0.8
    sectors = [frac(edge & (theta >= a) & (theta < a + np.pi / 4)) for a in np.linspace(-np.pi, np.pi, 9)[:-1]]
    n_fail = int(fail.sum())
    largest = 0
    if n_fail:
        n, _, stats, _ = cv2.connectedComponentsWithStats(fail.astype(np.uint8), connectivity=8)
        largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
    return {"fail": frac(np.ones_like(wafer)), "center": frac(r < 0.35), "mid": frac((r >= 0.35) & (r < 0.7)),
            "edge": frac(edge), "edge_sector_max": max(sectors), "edge_sector_min": min(sectors),
            "largest_share": largest / max(n_fail, 1)}


def _mock_probs(m: np.ndarray) -> np.ndarray:
    f = _features(m)
    score = dict.fromkeys(CLASSES, 0.02)
    if f["fail"] > 0.6:
        score["Near-full"] += 3
    elif f["fail"] > 0.25:
        score["Random"] += 2
    score["none"] += 3.0 * max(0.0, 0.08 - f["fail"]) / 0.08 + (0.6 if f["largest_share"] < 0.1 else 0)
    score["Center"] += 4 * max(0.0, f["center"] - 2 * f["edge"] - f["fail"])
    score["Donut"] += 4 * max(0.0, f["mid"] - f["center"] - f["edge"])
    score["Edge-Ring"] += 4 * max(0.0, f["edge_sector_min"] - f["center"])
    score["Edge-Loc"] += 4 * max(0.0, f["edge_sector_max"] - 2 * f["edge_sector_min"] - f["center"])
    score["Loc"] += 2 * max(0.0, f["largest_share"] - 0.3) * (f["fail"] < 0.25)
    score["Scratch"] += 1.5 * max(0.0, f["largest_share"] - 0.3) * (f["fail"] < 0.1) * (f["center"] < 0.2)
    s = np.array([score[c] for c in CLASSES])
    e = np.exp(2.5 * (s - s.max()))
    return e / e.sum()


def classify(wafer_map: np.ndarray) -> dict:
    m = validate_map(wafer_map)
    return result_dict(_mock_probs(m), _THETA_NONE, _TAU, MODEL_VERSION)


def classify_batch(maps: list[np.ndarray]) -> list[dict]:
    return [classify(m) for m in maps]


def explain(wafer_map: np.ndarray) -> dict:
    m = validate_map(wafer_map)
    pattern = classify(m)["pattern"]
    fail = (m == FAIL).astype(np.float32)
    heat = cv2.GaussianBlur(fail, (0, 0), sigmaX=max(1.0, min(m.shape) / 30)) * (m > 0)
    heat = heat / heat.max() if heat.max() > 0 else heat
    return {"heatmap": heat.astype(np.float32), "bbox": defect_bbox(m, heat, pattern), "pattern": pattern}


class WaferClassifier:
    """Same object API as wafer.infer.WaferClassifier."""
    model_version = MODEL_VERSION

    def classify(self, wafer_map):
        return classify(wafer_map)

    def classify_batch(self, maps):
        return classify_batch(maps)

    def explain(self, wafer_map):
        return explain(wafer_map)


def load_model(path: str = "models/mock", device: str = "cpu") -> WaferClassifier:
    return WaferClassifier()

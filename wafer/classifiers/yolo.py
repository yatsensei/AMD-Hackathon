"""Experimental YOLO-cls backend (Ultralytics classify).

Preserves the teammate preprocessing (nearest 128 + BGR render). Does **not** apply
CNN temperature / τ / θ. Confidence is raw softmax; review rules are not validated
(`needs_review` is always False). Never falls back to the CNN.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

from wafer.classifiers.base import ClassifierNotAvailableError
from wafer.classifiers.yolo_render import DEFAULT_SIZE, render
from wafer.constants import CLASSES
from wafer.env import REPO_ROOT

DEFAULT_WEIGHTS_DIR = REPO_ROOT / "models" / "yolo-cls-exp"
MODEL_VERSION = "yolo-cls-exp"
WHOLE_WAFER = {"Near-full", "Random"}

_UNAVAILABLE_HINTS = (
    "experimental YOLO-cls backend unavailable",
    "Install ultralytics and ensure models/yolo-cls-exp/best.pt is present.",
    "CNN is not substituted when YOLO is selected.",
)


def _where_on_wafer(cx: float, cy: float) -> str:
    """Plain-English location from a 0-1 position (heuristic, not a detector box)."""
    dx, dy = cx - 0.5, cy - 0.5
    r = (dx ** 2 + dy ** 2) ** 0.5
    if r < 0.15:
        return "centre"
    vert = "upper" if dy < -0.12 else "lower" if dy > 0.12 else ""
    horiz = "left" if dx < -0.12 else "right" if dx > 0.12 else ""
    side = "-".join(p for p in (vert, horiz) if p) or "middle"
    return f"{side} {'edge' if r > 0.35 else 'area'}"


def _densest_failure_spot(small: np.ndarray) -> tuple[float, float]:
    size = small.shape[0]
    valid = (small > 0).astype(np.float32)
    fail = (small == 2).astype(np.float32)
    k = max(3, size // 8) | 1
    density = cv2.blur(fail, (k, k)) / np.maximum(cv2.blur(valid, (k, k)), 1e-6)
    density[valid == 0] = 0
    hot = (density >= max(0.5, density[valid > 0].mean() + 0.25)).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(hot, connectivity=8)
    if n <= 1:
        y, x = np.unravel_index(int(density.argmax()), density.shape)
        return x / size, y / size
    biggest = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    cx, cy = cents[biggest]
    return float(cx / size), float(cy / size)


def _location(pattern: str, small: np.ndarray) -> str | None:
    if pattern == "none":
        return None
    if pattern in WHOLE_WAFER:
        return "whole wafer"
    if pattern == "Edge-Ring":
        return "all around the edge"
    return _where_on_wafer(*_densest_failure_spot(small))


def _ordered_names(names) -> list[str]:
    if isinstance(names, dict):
        return [names[i] for i in sorted(int(k) for k in names.keys())]
    return list(names)


class YoloClassifier:
    """Selectable experimental backend (`name="yolo"`). Requires ultralytics + weights."""

    name = "yolo"

    def __init__(self, weights_dir: str | Path | None = None, size: int | None = None):
        self.dir = Path(
            weights_dir
            or os.environ.get("WAFER_YOLO_DIR")
            or DEFAULT_WEIGHTS_DIR
        )
        self.cfg = self._load_config()
        self.size = int(size or self.cfg.get("imgsz") or DEFAULT_SIZE)
        self.model_version = str(self.cfg.get("version") or MODEL_VERSION)
        weights = self.dir / (self.cfg.get("weights") or "best.pt")
        if not weights.exists():
            raise ClassifierNotAvailableError(
                f"{_UNAVAILABLE_HINTS[0]}: missing weights {weights}. {_UNAVAILABLE_HINTS[1]} "
                f"{_UNAVAILABLE_HINTS[2]}"
            )
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ClassifierNotAvailableError(
                f"{_UNAVAILABLE_HINTS[0]}: ultralytics not installed. {_UNAVAILABLE_HINTS[1]} "
                f"{_UNAVAILABLE_HINTS[2]}"
            ) from e
        self.model = YOLO(str(weights))
        self._names = _ordered_names(self.model.names)
        if self._names != list(CLASSES):
            raise ClassifierNotAvailableError(
                f"YOLO class order {self._names!r} does not match wafer.constants.CLASSES "
                f"{list(CLASSES)!r}; refusing to serve mismatched labels"
            )
        # Cross-check committed metadata when present
        classes_json = self.dir / "classes.json"
        if classes_json.exists():
            meta = json.loads(classes_json.read_text())
            meta_names = [meta[str(i)] for i in range(len(CLASSES))]
            if meta_names != self._names:
                raise ClassifierNotAvailableError(
                    f"classes.json {meta_names!r} disagrees with weights {self._names!r}"
                )

    def _load_config(self) -> dict:
        path = self.dir / "config.yaml"
        if path.exists():
            return yaml.safe_load(path.read_text()) or {}
        return {"version": MODEL_VERSION, "imgsz": DEFAULT_SIZE, "weights": "best.pt"}

    def _result(self, wafer_map: np.ndarray, probs_vec: np.ndarray, pattern: str,
                confidence: float, top3: list[dict], small: np.ndarray) -> dict:
        probs = {c: float(probs_vec[i]) for i, c in enumerate(CLASSES)}
        # Argmax pattern from the probability vector (same order as CLASSES / weights).
        pred_idx = int(np.argmax(probs_vec))
        if CLASSES[pred_idx] != pattern:
            pattern = CLASSES[pred_idx]
            confidence = float(probs_vec[pred_idx])
        valid = small > 0
        fail_rate = float((small[valid] == 2).mean()) if valid.any() else 0.0
        return {
            "pattern": pattern,
            "confidence": float(confidence),
            "probs": probs,
            # No validated YOLO review policy — never invent CNN-style needs_review.
            "needs_review": False,
            "review_reasons": [],
            "model_version": self.model_version,
            "classifier": self.name,
            "defect_prob": float(1.0 - probs.get("none", 0.0)),
            "extras": {
                "experimental": True,
                "calibrated": False,
                "review_rules_validated": False,
                "confidence_note": (
                    "Uncalibrated raw softmax from YOLO-cls; CNN temperature / τ / θ were not applied."
                ),
                "training_disclosure": (
                    "Trained on a private stratified sample, not splits/wafer_split_v1.csv.gz; "
                    "overlaps the val demo stream. Not a fair comparison or held-out confirmation."
                ),
                "location": _location(pattern, small),
                "fail_rate": round(fail_rate, 3),
                "top3": top3,
            },
        }

    def classify(self, wafer_map: np.ndarray) -> dict:
        small, img = render(np.asarray(wafer_map), self.size)
        res = self.model.predict(img, imgsz=self.size, verbose=False)[0]
        names, probs = res.names, res.probs
        pattern = names[int(probs.top1)]
        confidence = float(probs.top1conf)
        vec = np.asarray(probs.data.float().cpu().numpy(), dtype=np.float64).reshape(-1)
        if vec.shape[0] != len(CLASSES):
            raise RuntimeError(f"YOLO returned {vec.shape[0]} probs; expected {len(CLASSES)}")
        top3 = [{"pattern": names[int(i)], "confidence": round(float(c), 3)}
                for i, c in zip(probs.top5[:3], probs.top5conf[:3].tolist())]
        return self._result(wafer_map, vec, pattern, confidence, top3, small)

    def classify_batch(self, maps: list[np.ndarray], batch_size: int = 32) -> list[dict]:
        out: list[dict] = []
        for i in range(0, len(maps), batch_size):
            chunk = maps[i:i + batch_size]
            rendered = [render(np.asarray(m), self.size) for m in chunk]
            imgs = [img for _, img in rendered]
            preds = self.model.predict(imgs, imgsz=self.size, verbose=False)
            for (small, _), res in zip(rendered, preds):
                names, probs = res.names, res.probs
                pattern = names[int(probs.top1)]
                confidence = float(probs.top1conf)
                vec = np.asarray(probs.data.float().cpu().numpy(), dtype=np.float64).reshape(-1)
                top3 = [{"pattern": names[int(i)], "confidence": round(float(c), 3)}
                        for i, c in zip(probs.top5[:3], probs.top5conf[:3].tolist())]
                out.append(self._result(None, vec, pattern, confidence, top3, small))
        return out

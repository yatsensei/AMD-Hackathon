"""Turn calibrated probabilities into the decision the API returns (spec 7.4 steps 4-6).

Shared by wafer.infer, wafer.infer_mock and the evaluation code, so the rule the agent sees
is exactly the rule that was scored.
"""
from __future__ import annotations

import numpy as np

from wafer.constants import CLASS_TO_IDX, CLASSES, CONFUSABLE_PAIRS, DEFECT_IDX, NONE_IDX

DEFAULT_MARGIN = 0.15
_CONFUSABLE = {frozenset((CLASS_TO_IDX[a], CLASS_TO_IDX[b])) for a, b in CONFUSABLE_PAIRS}
REVIEW_REASONS = ("low_confidence", "confusable_pair", "possible_defect")


def decide_batch(probs: np.ndarray, theta_none: float, tau: float, margin: float = DEFAULT_MARGIN) -> dict:
    """Vectorised decision for (N, 9) calibrated probabilities.

    Returns arrays: pred (N,), confidence (N,), needs_review (N,), and one bool array per reason.
    """
    p = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-p, axis=1, kind="stable")
    top1, top2 = order[:, 0], order[:, 1]
    defect_arg = np.asarray(DEFECT_IDX)[np.argmax(p[:, DEFECT_IDX], axis=1)]
    # `none` only if it is the top class AND p(none) >= theta: the operating point can turn
    # argmax-`none` wafers into defect calls (more recall), never the reverse.
    pred = np.where((top1 == NONE_IDX) & (p[:, NONE_IDX] >= theta_none), NONE_IDX, defect_arg)
    conf = p[np.arange(len(p)), pred]

    p1, p2 = p[np.arange(len(p)), top1], p[np.arange(len(p)), top2]
    pair = np.array([frozenset((a, b)) in _CONFUSABLE for a, b in zip(top1, top2)], dtype=bool)

    reasons = {
        "low_confidence": conf < tau,
        "confusable_pair": pair & (p1 - p2 < margin),
        "possible_defect": (pred != NONE_IDX) & (top1 == NONE_IDX),
    }
    needs_review = np.zeros(len(p), dtype=bool)
    for r in reasons.values():
        needs_review |= r
    return {"pred": pred, "confidence": conf, "needs_review": needs_review, "top1": top1, "top2": top2, **reasons}


def result_dict(probs: np.ndarray, theta_none: float, tau: float, model_version: str,
                margin: float = DEFAULT_MARGIN) -> dict:
    """One wafer's API output (spec 9.1). Plain Python types only, so it is JSON-serialisable."""
    p = np.asarray(probs, dtype=np.float64).reshape(1, -1)
    p = p / p.sum()
    d = decide_batch(p, theta_none, tau, margin)
    t1, t2 = int(d["top1"][0]), int(d["top2"][0])
    return {
        "pattern": CLASSES[int(d["pred"][0])],
        "confidence": float(d["confidence"][0]),
        "probs": {c: float(p[0, i]) for i, c in enumerate(CLASSES)},
        "top2": [[CLASSES[t1], float(p[0, t1])], [CLASSES[t2], float(p[0, t2])]],
        "defect_prob": float(1.0 - p[0, NONE_IDX]),
        "needs_review": bool(d["needs_review"][0]),
        "review_reasons": [r for r in REVIEW_REASONS if bool(d[r][0])],
        "model_version": model_version,
    }

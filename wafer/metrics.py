"""Every metric in spec Section 7.1 plus the statistics in 7.3.

Hot-path metrics (macro-F1 etc.) are computed from a 9x9 confusion matrix built with bincount,
so 1,000 lot-bootstrap resamples stay cheap. tests/test_metrics.py checks them against sklearn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wafer.constants import CLASS_TO_IDX, CLASSES, CONFUSABLE_PAIRS, DEFECT_IDX, NONE_IDX

K = len(CLASSES)


# --------------------------------------------------------------------------- confusion-based
def confusion(y, pred) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    y, pred = np.asarray(y), np.asarray(pred)
    return np.bincount(y * K + pred, minlength=K * K).reshape(K, K)


def _prf(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1).astype(float), cm.sum(0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return precision, recall, f1, support


def macro_f1(cm: np.ndarray, classes=None) -> float:
    """Mean F1 over `classes` (default all 9). Classes absent from both y and pred score 0, like
    sklearn with zero_division=0 and explicit labels."""
    f1 = _prf(cm)[2]
    idx = range(K) if classes is None else classes
    return float(np.mean([f1[i] for i in idx]))


def balanced_accuracy(cm: np.ndarray) -> float:
    _, recall, _, support = _prf(cm)
    present = support > 0
    return float(recall[present].mean()) if present.any() else 0.0


def defect_recall_precision(cm: np.ndarray) -> tuple[float, float]:
    d = DEFECT_IDX
    true_def = cm[d, :].sum()
    pred_def = cm[:, d].sum()
    hit = cm[np.ix_(d, d)].sum()                      # defect predicted as *some* defect
    return (float(hit / true_def) if true_def else 0.0, float(hit / pred_def) if pred_def else 0.0)


def mcc(cm: np.ndarray) -> float:
    t, p, c, s = cm.sum(1).astype(float), cm.sum(0).astype(float), float(np.trace(cm)), float(cm.sum())
    denom = np.sqrt((s**2 - (p**2).sum()) * (s**2 - (t**2).sum()))
    return float((c * s - (t * p).sum()) / denom) if denom > 0 else 0.0


def cohen_kappa(cm: np.ndarray) -> float:
    s = cm.sum()
    po = np.trace(cm) / s
    pe = (cm.sum(0) * cm.sum(1)).sum() / s**2
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def headline(cm: np.ndarray) -> dict:
    precision, recall, f1, support = _prf(cm)
    dr, dp = defect_recall_precision(cm)
    return {
        "macro_f1": macro_f1(cm),
        "defect_macro_f1": macro_f1(cm, DEFECT_IDX),
        "balanced_accuracy": balanced_accuracy(cm),
        "defect_recall": dr,
        "defect_precision": dp,
        "accuracy": float(np.trace(cm) / cm.sum()),
        "weighted_f1": float((f1 * support).sum() / support.sum()),
        "mcc": mcc(cm),
        "cohen_kappa": cohen_kappa(cm),
    }


def per_class(cm: np.ndarray, lots: dict | None = None) -> pd.DataFrame:
    precision, recall, f1, support = _prf(cm)
    df = pd.DataFrame({"class": CLASSES, "precision": precision, "recall": recall, "f1": f1,
                       "support": support.astype(int)})
    if lots is not None:
        df["support_lots"] = [lots.get(c, 0) for c in CLASSES]
    return df


def confusable_pair_rates(cm: np.ndarray) -> dict:
    """Share of the pair's wafers predicted as the other member of the pair."""
    out = {}
    for a, b in CONFUSABLE_PAIRS:
        i, j = CLASS_TO_IDX[a], CLASS_TO_IDX[b]
        n = cm[i].sum() + cm[j].sum()
        out[f"{a}<->{b}"] = float((cm[i, j] + cm[j, i]) / n) if n else 0.0
    return out


# --------------------------------------------------------------------------- probability-based
def nll(y, P) -> float:
    P = np.clip(np.asarray(P, dtype=np.float64), 1e-12, 1.0)
    return float(-np.log(P[np.arange(len(y)), y]).mean())


def brier(y, P) -> float:
    onehot = np.eye(K)[np.asarray(y)]
    return float(((np.asarray(P) - onehot) ** 2).sum(1).mean())


def top2_accuracy(y, P) -> float:
    top2 = np.argsort(-np.asarray(P), axis=1)[:, :2]
    return float((top2 == np.asarray(y)[:, None]).any(1).mean())


def ece(conf, correct, n_bins: int = 15) -> float:
    """Expected calibration error with equal-mass bins."""
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return float("nan")
    order = np.argsort(conf)
    total = 0.0
    for b in np.array_split(order, min(n_bins, len(conf))):
        total += len(b) * abs(correct[b].mean() - conf[b].mean())
    return float(total / len(conf))


def classwise_ece(y, P, n_bins: int = 15) -> float:
    y, P = np.asarray(y), np.asarray(P)
    return float(np.mean([ece(P[:, c], y == c, n_bins) for c in range(K)]))


def auc_scores(y, P) -> dict:
    """Macro one-vs-rest ROC-AUC and PR-AUC over classes with both positives and negatives."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, P = np.asarray(y), np.asarray(P)
    roc, pr = [], []
    for c in range(K):
        pos = y == c
        if 0 < pos.sum() < len(y):
            roc.append(roc_auc_score(pos, P[:, c]))
            pr.append(average_precision_score(pos, P[:, c]))
    return {"roc_auc_macro": float(np.mean(roc)) if roc else float("nan"),
            "pr_auc_macro": float(np.mean(pr)) if pr else float("nan")}


# --------------------------------------------------------------------------- selective prediction
def coverage_risk(conf, y, pred, coverages=np.linspace(0.5, 1.0, 11)) -> pd.DataFrame:
    """Accuracy and macro-F1 on the most-confident `coverage` share of wafers."""
    conf, y, pred = np.asarray(conf), np.asarray(y), np.asarray(pred)
    order = np.argsort(-conf, kind="stable")
    rows = []
    for c in coverages:
        k = max(1, int(round(c * len(order))))
        keep = order[:k]
        cm = confusion(y[keep], pred[keep])
        rows.append({"coverage": float(c), "accuracy": float(np.trace(cm) / k), "macro_f1": macro_f1(cm),
                     "min_confidence": float(conf[keep].min())})
    return pd.DataFrame(rows)


def aurc(conf, correct) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    order = np.argsort(-np.asarray(conf), kind="stable")
    err = 1.0 - np.asarray(correct, dtype=float)[order]
    return float((np.cumsum(err) / np.arange(1, len(err) + 1)).mean()) if len(err) else float("nan")


def selective_accuracy(conf, correct, coverage: float) -> float:
    order = np.argsort(-np.asarray(conf), kind="stable")
    k = max(1, int(round(coverage * len(order))))
    return float(np.asarray(correct, dtype=float)[order[:k]].mean()) if len(order) else float("nan")


def review_rates(y, needs_review) -> dict:
    y, r = np.asarray(y), np.asarray(needs_review, dtype=bool)
    return {c: (float(r[y == i].mean()) if (y == i).any() else float("nan")) for i, c in enumerate(CLASSES)}


# --------------------------------------------------------------------------- statistics (7.3)
def _group_index(groups) -> list[np.ndarray]:
    codes, _ = pd.factorize(np.asarray(groups))
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    return np.split(order, bounds)


def bootstrap_ci(fn, groups, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Resample groups (lots / split groups) with replacement; fn(idx) -> float on row indices."""
    parts = _group_index(groups)
    rng = np.random.default_rng(seed)
    point = fn(np.concatenate(parts))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(parts), size=len(parts))
        stats[b] = fn(np.concatenate([parts[i] for i in pick]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"point": float(point), "lo": float(lo), "hi": float(hi)}


def paired_bootstrap(fn_a, fn_b, groups, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """CI of fn_a - fn_b with the *same* group resample for both. Significant iff CI excludes 0."""
    res = bootstrap_ci(lambda idx: fn_a(idx) - fn_b(idx), groups, n_boot, seed, alpha)
    res["significant"] = bool(res["lo"] > 0 or res["hi"] < 0)
    return res


def mcnemar(y, pred_a, pred_b) -> dict:
    """Exact McNemar on wafer-level correctness. Secondary: ignores lot correlation (spec 7.3)."""
    from scipy.stats import binomtest
    ca, cb = np.asarray(pred_a) == np.asarray(y), np.asarray(pred_b) == np.asarray(y)
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"a_only_correct": b, "b_only_correct": c, "p_value": float(p)}


# --------------------------------------------------------------------------- everything
def evaluate_predictions(y, P, decision: dict, groups=None, n_boot: int = 0, seed: int = 0) -> dict:
    """All 7.1 metrics for calibrated probabilities P and a decide_batch() result. JSON-ready."""
    y, P = np.asarray(y), np.asarray(P)
    pred, conf, review = decision["pred"], decision["confidence"], decision["needs_review"]
    correct = pred == y
    cm = confusion(y, pred)
    pred_def = pred != NONE_IDX
    out = {
        "n": int(len(y)),
        **headline(cm),
        "argmax_macro_f1": macro_f1(confusion(y, P.argmax(1))),
        "nll": nll(y, P), "brier": brier(y, P), "top2_accuracy": top2_accuracy(y, P),
        **auc_scores(y, P),
        "ece": ece(P.max(1), P.argmax(1) == y),
        "ece_defect": ece(conf[pred_def], correct[pred_def]),
        "classwise_ece": classwise_ece(y, P),
        "aurc": aurc(conf, correct),
        "aurc_defect": aurc(conf[pred_def], correct[pred_def]),
        "selective_accuracy_defect_90": selective_accuracy(conf[pred_def], correct[pred_def], 0.90),
        "selective_accuracy_defect_95": selective_accuracy(conf[pred_def], correct[pred_def], 0.95),
        "review_rate": float(review.mean()),
        "review_rate_per_class": review_rates(y, review),
        "confusable_pair_error": confusable_pair_rates(cm),
        "confusion": cm.tolist(),
        "per_class": per_class(cm).to_dict(orient="records"),
    }
    if groups is not None and n_boot:
        def on(fn):
            return bootstrap_ci(lambda idx: fn(confusion(y[idx], pred[idx])), groups, n_boot, seed)
        out["ci"] = {
            "macro_f1": on(macro_f1),
            "defect_macro_f1": on(lambda cm_: macro_f1(cm_, DEFECT_IDX)),
            "balanced_accuracy": on(balanced_accuracy),
            "defect_recall": on(lambda cm_: defect_recall_precision(cm_)[0]),
            "per_class_f1": {c: on(lambda cm_, i=i: float(_prf(cm_)[2][i])) for i, c in enumerate(CLASSES)},
        }
    return out

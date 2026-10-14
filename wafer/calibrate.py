"""Calibration, operating point and abstention, all fitted on validation (spec 7.4).

1. combine members/TTA: average probabilities, z = log(p_avg)  (single model: z = logits)
2. prior correction:   z' = z + log pi_natural - log pi_train_eff   (exact, no fitting)
3. temperature:        p = softmax(z' / T), T fitted by minimising val NLL
4. operating point:    smallest theta_none with val defect recall >= recall_target (0 = argmax)
5. abstention:         smallest tau with selective accuracy >= sel_acc_target on predicted defects
"""
from __future__ import annotations

import numpy as np

from wafer.constants import CLASSES, NONE_IDX
from wafer.decision import DEFAULT_MARGIN, decide_batch

DEFAULT_RECALL_TARGET = 0.95
DEFAULT_SEL_ACC_TARGET = 0.95


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def log_mean_probs(member_logits: list[np.ndarray]) -> np.ndarray:
    """Step 1: average member (and TTA) probabilities, back to log space."""
    p = np.mean([softmax(z) for z in member_logits], axis=0)
    return np.log(np.clip(p, 1e-12, 1.0))


def prior_correct(z: np.ndarray, pi_natural, pi_train_eff) -> np.ndarray:
    pn = np.clip(np.asarray(pi_natural, dtype=np.float64), 1e-12, None)
    pe = np.clip(np.asarray(pi_train_eff, dtype=np.float64), 1e-12, None)
    shift = np.log(pn) - np.log(pe)
    return np.asarray(z, dtype=np.float64) + shift[None, :]


def fit_temperature(z: np.ndarray, y: np.ndarray, iters: int = 200) -> float:
    """Single temperature minimising NLL (LBFGS on log T, float64)."""
    import torch
    zt = torch.tensor(np.asarray(z), dtype=torch.float64)
    yt = torch.tensor(np.asarray(y), dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(zt / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def choose_theta_none(P: np.ndarray, y: np.ndarray, recall_target: float) -> float:
    """Smallest theta reaching `recall_target` defect recall under the rule "predict `none`
    only if it is the top class and p(none) >= theta" (decision.decide_batch).

    theta = 0 reproduces plain argmax; raising it only converts argmax-`none` wafers into defect
    calls. So if argmax already meets the target, theta = 0 and recall is never given away.
    """
    P, y = np.asarray(P), np.asarray(y)
    is_def = y != NONE_IDX
    n_def = int(is_def.sum())
    if n_def == 0:
        return 0.0
    missed = is_def & (P.argmax(1) == NONE_IDX)             # defects that argmax calls `none`
    need = int(np.ceil(recall_target * n_def)) - (n_def - int(missed.sum()))
    if need <= 0:
        return 0.0
    p_none = np.sort(P[missed, NONE_IDX])
    return float(np.nextafter(p_none[need - 1], np.inf))


def choose_tau(conf: np.ndarray, correct: np.ndarray, target: float) -> tuple[float, float]:
    """Smallest tau whose kept set {conf >= tau} reaches `target` accuracy -> (tau, coverage).
    If even the single most confident prediction misses, tau = 1.0 (review everything)."""
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return 1.0, 0.0
    order = np.argsort(-conf, kind="stable")
    cum_acc = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    ok = np.flatnonzero(cum_acc >= target)
    if len(ok) == 0:
        return 1.0, 0.0
    k = int(ok.max()) + 1
    tau = float(conf[order[k - 1]])
    return tau, float((conf >= tau).mean())


def calibrate(z: np.ndarray, y: np.ndarray, pi_natural, pi_train_eff,
              recall_target: float = DEFAULT_RECALL_TARGET, sel_acc_target: float = DEFAULT_SEL_ACC_TARGET,
              margin: float = DEFAULT_MARGIN) -> dict:
    """Fit steps 2-5 on validation outputs `z` (logits, or log_mean_probs for ensembles/TTA)."""
    y = np.asarray(y)
    zc = prior_correct(z, pi_natural, pi_train_eff)
    T = fit_temperature(zc, y)
    P = softmax(zc / T)
    theta = choose_theta_none(P, y, recall_target)
    d0 = decide_batch(P, theta, tau=0.0, margin=margin)
    pred_def = d0["pred"] != NONE_IDX
    tau, cov = choose_tau(d0["confidence"][pred_def], (d0["pred"] == y)[pred_def], sel_acc_target)
    return {
        "temperature": T,
        "theta_none": theta,
        "tau": tau,
        "margin": margin,
        "recall_target": recall_target,
        "sel_acc_target": sel_acc_target,
        "pi_natural": [float(v) for v in pi_natural],
        "pi_train_eff": [float(v) for v in pi_train_eff],
        "val_defect_coverage_at_tau": cov,
        "classes": list(CLASSES),
    }


def apply(cal: dict, z: np.ndarray) -> np.ndarray:
    """Calibrated probabilities for outputs z with a fitted calibration dict."""
    return softmax(prior_correct(z, cal["pi_natural"], cal["pi_train_eff"]) / cal["temperature"])


def decide(cal: dict, P: np.ndarray) -> dict:
    return decide_batch(P, cal["theta_none"], cal["tau"], cal.get("margin", DEFAULT_MARGIN))


def combine_members(outputs: list[np.ndarray], tta: bool) -> np.ndarray:
    """Step 1 exactly as served: one model without TTA -> its logits; otherwise the log of the
    averaged probabilities (outputs may be logits or TTA log-probabilities)."""
    if len(outputs) == 1 and not tta:
        return np.asarray(outputs[0], dtype=np.float64)
    return log_mean_probs(outputs)

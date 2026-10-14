"""Spec 11 test_metrics.py: metrics vs hand/sklearn values, calibration and decision rules."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, matthews_corrcoef

from wafer import calibrate, metrics
from wafer.constants import CLASS_TO_IDX, DEFECT_IDX, NONE_IDX
from wafer.decision import decide_batch, result_dict


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 9, 500)
    pred = np.where(rng.random(500) < 0.7, y, rng.integers(0, 9, 500))
    return y, pred


def test_confusion_metrics_match_sklearn(toy):
    y, pred = toy
    cm = metrics.confusion(y, pred)
    assert cm.sum() == len(y)
    assert np.isclose(metrics.macro_f1(cm), f1_score(y, pred, labels=range(9), average="macro", zero_division=0))
    assert np.isclose(metrics.macro_f1(cm, DEFECT_IDX),
                      f1_score(y, pred, labels=DEFECT_IDX, average="macro", zero_division=0))
    assert np.isclose(metrics.balanced_accuracy(cm), balanced_accuracy_score(y, pred))
    assert np.isclose(metrics.mcc(cm), matthews_corrcoef(y, pred))
    assert np.isclose(metrics.cohen_kappa(cm), cohen_kappa_score(y, pred))


def test_hand_computed_example():
    # 4 Loc, 4 none: 3 Loc right, 1 Loc -> none; 1 none -> Center
    L, N, C = CLASS_TO_IDX["Loc"], NONE_IDX, CLASS_TO_IDX["Center"]
    y = np.array([L, L, L, L, N, N, N, N])
    pred = np.array([L, L, L, N, N, N, N, C])
    cm = metrics.confusion(y, pred)
    recall, precision = metrics.defect_recall_precision(cm)
    assert recall == 0.75 and precision == 0.75                 # 3 of 4 defects; 3 of 4 defect calls
    f1 = metrics.per_class(cm).set_index("class")["f1"]
    assert np.isclose(f1["Loc"], 2 * 1 * 0.75 / 1.75) and np.isclose(f1["none"], 0.75)


def test_ece_zero_when_perfectly_calibrated():
    conf = np.repeat([0.25, 0.75], 400)
    correct = np.concatenate([np.arange(400) % 4 == 0, np.arange(400) % 4 != 0])
    assert metrics.ece(conf, correct, n_bins=2) < 1e-12


def test_bootstrap_ci_contains_point_and_paired_self_contains_zero(toy):
    y, pred = toy
    groups = np.repeat(np.arange(100), 5)

    def f(i):
        return metrics.macro_f1(metrics.confusion(y[i], pred[i]))
    ci = metrics.bootstrap_ci(f, groups, n_boot=200)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    d = metrics.paired_bootstrap(f, f, groups, n_boot=200)
    assert d["lo"] <= 0 <= d["hi"] and not d["significant"]


def test_prior_correction_identity_and_effect():
    z = np.random.default_rng(0).normal(size=(5, 9))
    pi = np.full(9, 1 / 9)
    assert np.allclose(calibrate.prior_correct(z, pi, pi), z)
    nat = np.array([0.01] * 8 + [0.92])
    zc = calibrate.prior_correct(np.zeros((1, 9)), nat, pi)
    assert calibrate.softmax(zc)[0].argmax() == NONE_IDX


def test_temperature_recovers_known_scale():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(4000, 9)) * 2
    y = np.array([rng.choice(9, p=p) for p in calibrate.softmax(true_logits)])
    T = calibrate.fit_temperature(true_logits * 3, y)          # over-confident by 3x
    assert 2.5 < T < 3.5


def test_theta_none_gives_stated_recall():
    rng = np.random.default_rng(1)
    y = np.where(rng.random(2000) < 0.2, rng.choice(DEFECT_IDX, 2000), NONE_IDX)
    P = rng.dirichlet(np.ones(9), 2000)
    theta = calibrate.choose_theta_none(P, y, 0.95)
    d = decide_batch(P, theta, tau=0)
    cm = metrics.confusion(y, d["pred"])
    assert metrics.defect_recall_precision(cm)[0] >= 0.95
    assert theta > 0
    d2 = decide_batch(P, theta * 0.99, tau=0)                 # any smaller theta misses the target
    assert metrics.defect_recall_precision(metrics.confusion(y, d2["pred"]))[0] < 0.95 + 1e-9


def test_theta_none_never_gives_away_recall_argmax_already_has():
    """If plain argmax already meets the target, theta = 0 and decisions equal argmax."""
    rng = np.random.default_rng(3)
    y = np.where(rng.random(1000) < 0.2, rng.choice(DEFECT_IDX, 1000), NONE_IDX)
    P = np.full((1000, 9), 0.02)
    P[np.arange(1000), y] = 0.84                               # perfect, confident model
    assert calibrate.choose_theta_none(P, y, 0.95) == 0.0
    d = decide_batch(P, 0.0, tau=0)
    assert (d["pred"] == P.argmax(1)).all()


def test_tau_reaches_target_with_max_coverage():
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    correct = np.array([1, 1, 0, 1, 0])
    tau, cov = calibrate.choose_tau(conf, correct, 0.75)
    assert tau == 0.6 and cov == 0.8                            # top 4: 3/4 correct


def test_decision_reasons():
    p = np.full(9, 0.01)
    p[NONE_IDX], p[CLASS_TO_IDX["Loc"]], p[CLASS_TO_IDX["Center"]] = 0.45, 0.25, 0.23
    p /= p.sum()
    r = result_dict(p, theta_none=0.6, tau=0.5, model_version="t")
    assert r["pattern"] == "Loc"                                # p(none) below theta -> best defect
    assert set(r["review_reasons"]) == {"low_confidence", "possible_defect"}
    q = np.full(9, 0.01)
    q[CLASS_TO_IDX["Edge-Loc"]], q[CLASS_TO_IDX["Scratch"]] = 0.48, 0.45
    q /= q.sum()
    r2 = result_dict(q, theta_none=0.5, tau=0.1, model_version="t")
    assert r2["pattern"] == "Edge-Loc" and r2["review_reasons"] == ["confusable_pair"]

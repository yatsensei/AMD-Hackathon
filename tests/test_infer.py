"""Spec 11 test_infer.py. A tiny model is trained on synthetic data (CPU, seconds), exported,
and served; the same contract tests also run against wafer.infer_mock."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.synthetic import circle_wafer, draw_pattern, make_raw_df
from wafer import calibrate, infer_mock
from wafer.constants import CLASSES
from wafer.data.dataset import WaferData
from wafer.data.pipeline import main as prepare_main
from wafer.env import REPO_ROOT
from wafer.predict import predict_rows

KEYS = {"pattern": str, "confidence": float, "probs": dict, "top2": list, "defect_prob": float,
        "needs_review": bool, "review_reasons": list, "model_version": str}


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """(data_dir, model_dir) for a 32px small_cnn trained for a few steps on synthetic wafers."""
    from wafer import export, train
    d = tmp_path_factory.mktemp("e2e")
    make_raw_df().to_pickle(d / "LSWMD.pkl")
    prepare_main(["--raw", str(d / "LSWMD.pkl"), "--interim-dir", str(d / "interim"), "--out-dir", str(d / "proc"),
                  "--report", str(d / "report.md"), "--sizes", "32", "--split-seeds", "3", "--jobs", "1",
                  "--max-rel-dev", "10"] + sum((["--min-count", f"{c}=2"] for c in CLASSES), []))
    cfg = {"experiment": "T", "model": "small_cnn", "encoding": "area", "size": 32, "loss": "weighted_ce",
           "sampler": "none_downsample", "none_per_epoch": 300, "seeds": [0], "amp": False,
           "optim": {"lr": 3e-3, "wd": 1e-4, "epochs": 3, "batch_size": 64, "warmup_epochs": 1}}
    (d / "t.yaml").write_text(yaml.safe_dump(cfg))
    runs = train.main(["--config", str(d / "t.yaml"), "--data-dir", str(d / "proc"), "--out", str(d / "exp"),
                       "--device", "cpu"])
    model_dir = export.export(runs, "vtest", d / "models", d / "proc", tta=False, device="cpu")
    return d / "proc", model_dir, runs[0]


@pytest.fixture(scope="session")
def clf(trained):
    from wafer.infer import load_model
    return load_model(trained[1], "cpu")


@pytest.fixture(params=["real", "mock"])
def api(request, clf):
    return clf if request.param == "real" else infer_mock.load_model()


def _check_result(r):
    assert set(r) == set(KEYS)
    for k, t in KEYS.items():
        assert isinstance(r[k], t), (k, type(r[k]))
    assert r["pattern"] in CLASSES and list(r["probs"]) == CLASSES
    assert abs(sum(r["probs"].values()) - 1) < 1e-5
    assert np.isclose(r["confidence"], r["probs"][r["pattern"]])
    assert np.isclose(r["defect_prob"], 1 - r["probs"]["none"])
    assert r["needs_review"] == bool(r["review_reasons"])
    json.dumps(r)


@pytest.mark.parametrize("shape", [(10, 10), (64, 64), (300, 200)])
def test_classify_contract(api, shape):
    rng = np.random.default_rng(0)
    m, _ = circle_wafer(*shape)
    m[(m == 1) & (rng.random(shape) < 0.05)] = 2
    _check_result(api.classify(m))


def test_classify_batch_matches_single(api):
    rng = np.random.default_rng(0)
    maps = [draw_pattern(c, 30, 34, rng) for c in CLASSES]
    batch = api.classify_batch(maps)
    for m, b in zip(maps, batch):
        s = api.classify(m)
        assert s["pattern"] == b["pattern"] and np.isclose(s["confidence"], b["confidence"], atol=1e-6)


@pytest.mark.parametrize("bad", [np.zeros((5, 5), np.uint8), np.full((5, 5), 3), np.zeros(5), np.zeros((0, 4)),
                                 np.array([[1, 1.5]]), np.array([["a"]], dtype=object)])
def test_invalid_input_raises_value_error(api, bad):
    with pytest.raises(ValueError):
        api.classify(bad)


def test_explain_contract(api):
    rng = np.random.default_rng(2)
    m = draw_pattern("Loc", 45, 48, rng)
    e = api.explain(m)
    assert e["heatmap"].shape == m.shape and e["heatmap"].dtype == np.float32
    assert e["heatmap"].min() >= 0 and e["heatmap"].max() <= 1
    if e["bbox"] is not None:
        x0, y0, x1, y1 = e["bbox"]
        assert 0 <= x0 <= x1 < m.shape[1] and 0 <= y0 <= y1 < m.shape[0]
        assert (m[y0:y1 + 1, x0:x1 + 1] > 0).any()
    else:
        assert e["pattern"] == "none"


def test_train_serve_parity(trained, clf):
    """classify(raw map) == the evaluation pipeline's calibrated probabilities (spec 11)."""
    import pickle
    data_dir, model_dir, _ = trained
    raw = pickle.loads((data_dir.parent / "interim" / "labeled.pkl").read_bytes()).set_index("uid")
    data = WaferData(data_dir / "wafers_s32.npz", data_dir / "metadata.parquet", encoding="area")
    va = data.indices("val")[:200]
    z = calibrate.combine_members([predict_rows(m, data, va) for m in clf.members], tta=False)
    P_eval = calibrate.apply(clf.cal, z)
    P_serve = np.array([[r["probs"][c] for c in CLASSES]
                        for r in clf.classify_batch([raw.at[int(u), "waferMap"] for u in data.uid[va]])])
    assert np.abs(P_eval - P_serve).max() < 1e-5


def test_latency_smoke(clf):
    from wafer.bench import latency_ms
    assert latency_ms(clf.classify, n=50) < 150


def test_export_artifacts(trained):
    _, model_dir, _ = trained
    for f in ["config.yaml", "calibration.json", "classes.json", "metrics_val.json", "README.md", "weights/m0.safetensors"]:
        assert (model_dir / f).exists(), f
    cal = json.loads((model_dir / "calibration.json").read_text())
    assert cal["temperature"] > 0 and 0 <= cal["theta_none"] and 0 <= cal["tau"] <= 1


def test_train_outputs(trained):
    _, _, run = trained
    for f in ["config.yaml", "weights.safetensors", "history.csv", "val_logits.npy", "val_uids.npy",
              "train_prior.json", "calibration.json", "val_metrics.json", "summary.json", "env.json"]:
        assert (run / f).exists(), f
    s = json.loads((run / "summary.json").read_text())
    assert s["cpu_ms"] > 0 and s["epochs"] == 3


def test_evaluate_val_and_final_guard(trained, tmp_path, monkeypatch):
    from wafer import evaluate
    data_dir, model_dir, _ = trained
    out = evaluate.run(model_dir, data_dir, final=False, comparators=[model_dir], out=tmp_path / "val",
                       device="cpu", n_boot=30)
    res = json.loads((out / "metrics.json").read_text())
    assert res["split"] == "val" and "ci" in res and res["sanity_inputs"]["empty_map"].startswith("ValueError")
    comp = res["comparators"][str(model_dir)]
    assert comp["delta_macro_f1"]["point"] == 0 and not comp["delta_macro_f1"]["significant"]
    for f in ["confusion_norm.png", "reliability_after.png", "coverage_risk.png", "slices.csv", "per_class.csv"]:
        assert (out / f).exists(), f

    final = tmp_path / "final"
    monkeypatch.setattr(evaluate, "FINAL_DIR", final)
    evaluate.run(model_dir, data_dir, final=True, comparators=[], out=None, device="cpu", n_boot=10)
    assert json.loads((final / "metrics.json").read_text())["split"] == "test"
    with pytest.raises(SystemExit, match="already been scored once"):
        evaluate.run(model_dir, data_dir, final=True, comparators=[], out=None, device="cpu", n_boot=10)


# ---------------------------------------------------------------- behaviour of a real exported model
REAL = REPO_ROOT / "models" / "v0" / "config.yaml"


@pytest.mark.skipif(not REAL.exists(), reason="no models/v0 yet (scripts/amd/run_e0.sh)")
def test_real_model_sanity_inputs():
    from wafer.infer import load_model
    clf = load_model(REAL.parent)
    yy, xx = np.mgrid[:52, :52]
    circle = (np.hypot(yy - 25.5, xx - 25.5) <= 26).astype(np.uint8)
    assert clf.classify(circle)["pattern"] == "none"
    assert clf.classify(circle * 2)["pattern"] == "Near-full"


@pytest.mark.skipif(not REAL.exists(), reason="no models/v0 yet (scripts/amd/run_e0.sh)")
def test_real_model_latency_budget():
    from wafer.bench import latency_ms
    from wafer.infer import load_model
    assert latency_ms(load_model(REAL.parent).classify) < 50


def test_module_level_functions_use_default_model(trained, monkeypatch):
    import wafer.infer as inf
    monkeypatch.setenv("WAFER_MODEL_DIR", str(trained[1]))
    monkeypatch.setattr(inf, "_default", None)
    m, _ = circle_wafer(30, 30)
    _check_result(inf.classify(m))
    assert inf.classify(m)["model_version"] == "vtest"
    assert Path(inf.default_model_dir()) == trained[1]

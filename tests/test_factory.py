"""Phase 4 tests: stream, alert rule, root causes, impact. Synthetic tables only (no data build)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from wafer.constants import CLASSES
from wafer.factory import (build_stream, check_alert, estimate_impact, load_root_causes, lookup_root_cause,
                           lots_caught_early)
from wafer.factory.stream import STREAM_COLUMNS


def fake_split(n_none=900, n_ring=400, n_other=100, seed=0):
    rng = np.random.default_rng(seed)
    labels = ["none"] * n_none + ["Edge-Ring"] * n_ring + list(rng.choice(["Center", "Loc", "Scratch"], n_other))
    n = len(labels)
    return pd.DataFrame({"uid": np.arange(n), "lotName": [f"lot{i // 25}" for i in range(n)],
                         "waferIndex": [i % 25 + 1 for i in range(n)], "label": labels})


def events(rows):
    """rows: (tool, pattern, hours after t0, lot) -> minimal stream for check_alert."""
    t0 = pd.Timestamp("2026-10-12 00:00")
    return pd.DataFrame([{"tool_id": t, "pattern": p, "timestamp": t0 + pd.Timedelta(hours=h), "lot_id": lot,
                          "uid": i} for i, (t, p, h, lot) in enumerate(rows)])


# --------------------------------------------------------------------------- stream
def test_stream_shape_and_scenario():
    src = fake_split()
    s = build_stream(wafers=src, n_lots=20, excursion_start_lot=10)
    assert list(s.columns) == STREAM_COLUMNS and len(s) == 500
    assert s.groupby("lot_id").size().eq(25).all()
    assert s.uid.is_unique and set(s.uid) <= set(src.uid)          # every wafer traces back
    exc = s[s.lot_id >= "LOT-010"]
    assert (exc.tool_id == "ETCH-04").all() and exc.injected.all() and (exc.label == "Edge-Ring").all()
    early = s[s.lot_id < "LOT-010"]
    assert not early.injected.any() and early.tool_id.nunique() == 4
    assert s.timestamp.is_monotonic_increasing


def test_stream_keeps_true_labels_and_is_deterministic():
    src = fake_split()
    a, b = build_stream(wafers=src, seed=3), build_stream(wafers=src, seed=3)
    pd.testing.assert_frame_equal(a, b)
    merged = a.merge(src, on="uid", suffixes=("", "_src"))
    assert (merged.label == merged.label_src).all() and (merged.lotName == merged.lotName_src).all()


def test_stream_partial_excursion_and_errors():
    s = build_stream(wafers=fake_split(), excursion_rate=0.4, seed=1)
    frac = s[s.lot_id >= "LOT-010"].injected.mean()
    assert 0.25 < frac < 0.55
    with pytest.raises(ValueError, match="need"):
        build_stream(wafers=fake_split(n_ring=10))
    with pytest.raises(ValueError, match="excursion_tool"):
        build_stream(wafers=fake_split(), excursion_tool="CMP-01")


def test_baseline_defect_rate():
    src = fake_split(n_none=2000, n_ring=300, n_other=600)
    healthy = lambda s: s[s.lot_id < "LOT-010"]
    clean = healthy(build_stream(wafers=src, baseline_defect_rate=0.0))
    assert (clean.label == "none").all()
    natural = healthy(build_stream(wafers=src, baseline_defect_rate=None, seed=2))
    assert (natural.label != "none").mean() > 0.15
    with pytest.raises(ValueError, match="baseline_defect_rate"):
        build_stream(wafers=src, baseline_defect_rate=1.5)


def test_stream_refuses_train():
    with pytest.raises(ValueError, match="trained on train"):
        build_stream("train")


# --------------------------------------------------------------------------- alerts
def test_alert_fires_on_third_wafer_within_window():
    df = events([("ETCH-04", "Edge-Ring", 0, "L1"), ("ETCH-04", "Edge-Ring", 10, "L1"),
                 ("ETCH-04", "Edge-Ring", 20, "L2"), ("ETCH-04", "Edge-Ring", 21, "L2")])
    [a] = check_alert(df)
    assert a["tool_id"] == "ETCH-04" and a["pattern"] == "Edge-Ring" and a["count_in_window"] == 3
    assert a["fired_at"] == "2026-10-12T20:00:00" and a["lot_ids"] == ["L1", "L2"]
    json.dumps(a)


def test_alert_respects_window_tool_pattern_and_none():
    spread = events([("T1", "Loc", 0, "L"), ("T1", "Loc", 12, "L"), ("T1", "Loc", 24, "L")])  # 24h apart: not within
    assert check_alert(spread) == []
    mixed_tools = events([("T1", "Loc", 0, "L"), ("T2", "Loc", 1, "L"), ("T3", "Loc", 2, "L")])
    assert check_alert(mixed_tools) == []
    mixed_patterns = events([("T1", "Loc", 0, "L"), ("T1", "Center", 1, "L"), ("T1", "Scratch", 2, "L")])
    assert check_alert(mixed_patterns) == []
    nones = events([("T1", "none", h, "L") for h in range(5)])
    assert check_alert(nones) == []
    assert len(check_alert(spread, window_hours=25)) == 1
    assert check_alert(mixed_tools.assign(tool_id="T1"), threshold=4) == []


def test_one_alert_per_tool_lists_later_patterns():
    df = events([("ETCH-04", "Edge-Ring", h, "L1") for h in (0, 1, 2)]
                + [("ETCH-04", "Edge-Loc", h, "L2") for h in (5, 6, 7)]
                + [("ETCH-01", "Loc", h, "L3") for h in (8, 9, 10)])
    first, other = check_alert(df)
    assert (first["tool_id"], first["pattern"]) == ("ETCH-04", "Edge-Ring")
    assert first["also_seen"] == [{"pattern": "Edge-Loc", "fired_at": "2026-10-12T07:00:00",
                                   "fired_lot_id": "L2", "count_in_window": 3}]
    assert (other["tool_id"], other["also_seen"]) == ("ETCH-01", [])
    assert len(check_alert(df, one_per_tool=False)) == 3


def test_alert_reads_prediction_not_label():
    df = events([("T1", "none", h, "L") for h in range(3)]).assign(label="Edge-Ring")
    assert check_alert(df) == []
    with pytest.raises(ValueError, match="missing"):
        check_alert(df.drop(columns="pattern"))


def test_lots_caught_early_counts_later_lots_on_tool():
    s = build_stream(wafers=fake_split(n_ring=275), n_lots=20, excursion_start_lot=10)   # no Edge-Ring left for baseline
    s = s.assign(pattern=s.label)                     # perfect classifier stand-in
    a = next(x for x in check_alert(s) if x["tool_id"] == "ETCH-04" and x["pattern"] == "Edge-Ring")
    assert a["fired_lot_id"] == "LOT-010"
    assert lots_caught_early(s, a) == 10              # LOT-011..LOT-020


# --------------------------------------------------------------------------- root causes / impact
def test_root_causes_cover_all_classes():
    table = load_root_causes()
    assert set(table) == set(CLASSES)
    for entry in table.values():
        assert {"cause", "machine", "source"} <= set(entry) and all(entry[k] for k in ("cause", "machine", "source"))
    er = lookup_root_cause("Edge-Ring")
    assert er["tool_family"] == "ETCH" and er["is_possibility"] is True
    assert "source_warnings" in er
    with pytest.raises(ValueError):
        lookup_root_cause("Edge-ring")


def test_estimate_impact():
    r = estimate_impact(9)
    assert r["subsequent_lots_observed"] == 9
    assert r["subsequent_lot_exposure"]["wafers"] == 225
    assert r["subsequent_lot_exposure"]["value_usd"] == 4_500_000.0
    assert r["value_at_risk_usd"] == 4_500_000.0
    assert r["hypothetical"] is False and r["savings_claimed"] is False
    assert estimate_impact(2, wafer_value=5000, wafers_per_lot=10)["subsequent_lot_exposure"]["value_usd"] == 100_000.0
    json.dumps(r)
    with pytest.raises(ValueError):
        estimate_impact(-1)


def test_early_alert_zero_subsequent_is_not_zero_total_or_savings():
    """Regression: as_of at fire time → 0 subsequent lots ≠ $0 total risk or money saved."""
    from wafer.factory.impact import impact_from_alert

    df = events([("ETCH-04", "Edge-Ring", 0, "L1"),
                 ("ETCH-04", "Edge-Ring", 1, "L1"),
                 ("ETCH-04", "Edge-Ring", 2, "L1")])
    [a] = check_alert(df)
    as_of = a["fired_at"]  # no lot has started after the alert yet
    r = impact_from_alert(df, a, as_of=as_of, wafer_value=20_000, wafers_per_lot=25)

    assert r["counts_what"] == "subsequent_lots_observed_since_alert"
    assert r["subsequent_lots_observed"] == 0
    assert "subsequent_lot_exposure" not in r
    assert r["value_at_risk_usd"] is None
    assert r["dollars_saved_usd"] is None
    assert r["intervention_recorded"] is False
    assert r["savings_claimed"] is False
    assert "not that total value at risk is zero" in r["counts_note"].lower()
    assert "saved money" in r["intervention_note"].lower() or "stopped production" in r["intervention_note"].lower()
    aff = r["affected_lot_exposure"]
    assert aff["lots"] == 1 and aff["lot_ids"] == ["L1"]
    assert aff["observed_wafers"] == 3
    assert aff["assumed_full_lot_wafers"] == 25
    assert aff["partial_lot"] is True
    assert "future_hypothetical_exposure" not in r
    json.dumps(r)


def test_subsequent_lots_after_alert_get_exposure_block():
    from wafer.factory.impact import impact_from_alert

    df = events([("ETCH-04", "Edge-Ring", 0, "L1"),
                 ("ETCH-04", "Edge-Ring", 1, "L1"),
                 ("ETCH-04", "Edge-Ring", 2, "L1"),
                 ("ETCH-04", "none", 4, "L2"),   # later lot on same tool (pattern ignored for count helper)
                 ("ETCH-04", "none", 5, "L3")])
    [a] = check_alert(df)
    r = impact_from_alert(df, a, as_of="2026-10-13", wafer_value=1000, wafers_per_lot=10)
    assert r["subsequent_lots_observed"] == 2
    exp = r["subsequent_lot_exposure"]
    assert exp["lots"] == 2
    assert exp["observed_wafers"] == 2  # one wafer each in L2/L3
    assert exp["assumed_full_lot_wafers"] == 20
    assert exp["observed_value_usd"] == 2000.0
    assert exp["partial_lot"] is True
    assert r["value_at_risk_usd"] == 2000.0  # observed, not assumed full
    assert r["savings_claimed"] is False



def test_add_predictions_feeds_check_alert():
    from tests.synthetic import draw_pattern
    from wafer.classifiers.cnn import CnnClassifier
    from wafer import infer_mock
    from wafer.factory import add_predictions
    src = fake_split()
    label = dict(zip(src.uid, src.label))
    rng = np.random.default_rng(0)
    s = build_stream(wafers=src, n_lots=12, excursion_start_lot=10)
    backend = CnnClassifier(classify_fn=infer_mock.classify, classify_batch_fn=infer_mock.classify_batch)
    out = add_predictions(s, classifier=backend,
                          get_map=lambda uid: draw_pattern(label[uid], 40, 40, rng))
    assert {"pattern", "confidence", "needs_review", "review_reasons",
            "model_version", "classifier"} <= set(out.columns)
    assert set(out.pattern) <= set(CLASSES) and (out.model_version == "mock").all()
    assert (out.classifier == "cnn").all()
    assert isinstance(check_alert(out), list)

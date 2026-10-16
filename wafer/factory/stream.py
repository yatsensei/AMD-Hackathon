"""Simulated factory stream (spec 9.3): real val/test wafers regrouped into fictional lots on tools.

    stream = build_stream("val")              # lot_id, tool_id, timestamp, uid, lotName, waferIndex, label, injected
    stream = add_predictions(stream)          # + pattern, confidence, ... from wafer.infer.classify_batch

Demo scenario: from lot `excursion_start_lot` onward every lot runs on `excursion_tool` and
(`excursion_rate` of) its wafers are real wafers whose true label is `excursion_pattern`. The
maps are untouched, so the alert only fires if the classifier actually sees the pattern.
Stream "val" while building the demo; switch to "test" only after the final test evaluation.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from wafer.data.access import wafers_for_stream

TOOLS = ("ETCH-01", "ETCH-02", "ETCH-03", "ETCH-04")
STREAM_COLUMNS = ["lot_id", "wafer_in_lot", "tool_id", "timestamp", "uid", "lotName", "waferIndex",
                  "label", "injected"]


def build_stream(split: str = "val", n_lots: int = 20, lot_size: int = 25, tools: tuple[str, ...] = TOOLS,
                 excursion_tool: str = "ETCH-04", excursion_start_lot: int = 10,
                 excursion_pattern: str = "Edge-Ring", excursion_rate: float = 1.0,
                 baseline_defect_rate: float | None = 0.02,
                 start: str = "2026-10-12 06:00", lot_interval_hours: float = 2.0,
                 wafer_interval_minutes: float = 3.0, seed: int = 0,
                 wafers: pd.DataFrame | None = None) -> pd.DataFrame:
    """Lots are numbered from 1; lots 1..excursion_start_lot-1 rotate over `tools` (healthy baseline).

    Baseline wafers are drawn without replacement, disjoint from the injected wafers. A share
    `baseline_defect_rate` of them are defects (all 8 patterns in their split proportions), the
    rest `none`. None keeps the split's natural mix, ~15% defective: at that rate the 3-in-24h
    rule fires on healthy tools before the excursion in 50/50 seeds (labelled WM-811K is
    defect-enriched). `wafers` overrides the source table (tests); it needs uid, lotName,
    waferIndex and label.
    """
    if not 0.0 <= excursion_rate <= 1.0 or not 0.0 <= (baseline_defect_rate or 0.0) <= 1.0:
        raise ValueError("excursion_rate and baseline_defect_rate must be in [0, 1]")
    if excursion_tool not in tools:
        raise ValueError(f"excursion_tool {excursion_tool!r} not in tools {tools}")
    src = wafers_for_stream(split) if wafers is None else wafers
    rng = np.random.default_rng(seed)

    n_exc_lots = max(0, n_lots - excursion_start_lot + 1)
    inject = rng.random((n_exc_lots, lot_size)) < excursion_rate
    exc_pool = src[src["label"] == excursion_pattern]
    if inject.sum() > len(exc_pool):
        raise ValueError(f"need {int(inject.sum())} {excursion_pattern} wafers, split has {len(exc_pool)}")
    exc = exc_pool.iloc[rng.permutation(len(exc_pool))[:int(inject.sum())]]
    base_pool = src.drop(index=exc.index)
    n_base = n_lots * lot_size - int(inject.sum())
    if baseline_defect_rate is None:
        base = _sample(base_pool, n_base, rng)
    else:
        n_def = int(rng.binomial(n_base, baseline_defect_rate))
        is_none = base_pool["label"] == "none"
        base = _sample(pd.concat([_sample(base_pool[~is_none], n_def, rng),
                                  _sample(base_pool[is_none], n_base - n_def, rng)]), n_base, rng)

    rows, e, b = [], 0, 0
    t0 = pd.Timestamp(start)
    for lot in range(1, n_lots + 1):
        excursion = lot >= excursion_start_lot
        tool = excursion_tool if excursion else tools[(lot - 1) % len(tools)]
        lot_start = t0 + pd.Timedelta(hours=lot_interval_hours * (lot - 1))
        for w in range(lot_size):
            injected = bool(excursion and inject[lot - excursion_start_lot, w])
            if injected:
                r, e = exc.iloc[e], e + 1
            else:
                r, b = base.iloc[b], b + 1
            rows.append({"lot_id": f"LOT-{lot:03d}", "wafer_in_lot": w + 1, "tool_id": tool,
                         "timestamp": lot_start + pd.Timedelta(minutes=wafer_interval_minutes * w),
                         "uid": int(r["uid"]), "lotName": r["lotName"], "waferIndex": int(r["waferIndex"]),
                         "label": r["label"], "injected": injected})
    return pd.DataFrame(rows, columns=STREAM_COLUMNS)


def _sample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n > len(df):
        raise ValueError(f"split has too few wafers: need {n}, have {len(df)}")
    return df.iloc[rng.permutation(len(df))[:n]]


def add_predictions(stream: pd.DataFrame, classify_batch: Callable | None = None,
                    get_map: Callable | None = None) -> pd.DataFrame:
    """Add the classifier's view of each wafer: the columns check_alert and the agent read.

    Defaults to wafer.infer.classify_batch on maps from data/interim/labeled.pkl
    (`python scripts/prepare_data.py`). Pass wafer.infer_mock.classify_batch to run without torch.
    """
    if classify_batch is None:
        from wafer.infer import classify_batch
    if get_map is None:
        from wafer.data.access import get_wafer_map as get_map
    results = classify_batch([get_map(u) for u in stream["uid"]])
    out = stream.copy()
    out["pattern"] = [r["pattern"] for r in results]
    out["confidence"] = [round(r["confidence"], 4) for r in results]
    out["defect_prob"] = [round(r["defect_prob"], 4) for r in results]
    out["needs_review"] = [r["needs_review"] for r in results]
    out["review_reasons"] = [";".join(r["review_reasons"]) for r in results]
    out["model_version"] = [r["model_version"] for r in results]
    return out

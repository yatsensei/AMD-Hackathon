"""Line-level views for the agent: overview, trends, wafer search, decisions, explanations.

Pure functions over the scored stream (everything is cut at `as_of`, so the replay never
leaks future wafers). The agent tools in wafer.agent.tools wrap these.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

from wafer.constants import CLASSES, CONFUSABLE_PAIRS, FAIL, NO_DIE
from wafer.factory.alerts import check_alert
from wafer.factory.history import has_predictions, stream_as_of

# Labelled WM-811K counts (172,950 labelled maps; splits/wafer_split_v1.csv.gz).
DATASET_COUNTS = {"none": 147431, "Edge-Ring": 9680, "Edge-Loc": 5189, "Center": 4294, "Loc": 3593,
                  "Scratch": 1193, "Random": 866, "Donut": 555, "Near-full": 149}

PATTERN_LOOKS = {
    "Center": "A cluster of failing dies around the middle of the wafer.",
    "Donut": "A ring of failing dies part-way out from the centre, with a passing centre and edge.",
    "Edge-Loc": "A localised cluster of failing dies at one part of the wafer edge, not the whole rim.",
    "Edge-Ring": "Failing dies all the way (or most of the way) around the wafer edge.",
    "Loc": "A localised cluster of failing dies somewhere away from the centre and the edge.",
    "Near-full": "Almost every die on the wafer fails.",
    "Random": "Failing dies scattered across the wafer with no spatial structure.",
    "Scratch": "A thin line or arc of failing dies, often curved, crossing part of the wafer.",
    "none": "No spatial defect pattern: any failing dies are sparse background.",
}


def _defect(df: pd.DataFrame) -> pd.Series:
    return df["pattern"].astype(str) != "none"


def _round(x: float, n: int = 3) -> float:
    return float(round(float(x), n))


def _lot_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per lot: tool, start, wafers, defects, top defect pattern."""
    d = df.assign(_d=_defect(df), _t=pd.to_datetime(df["timestamp"]))
    g = d.groupby("lot_id", sort=False)
    out = pd.DataFrame({
        "tool_id": g["tool_id"].first(),
        "start": g["_t"].min(),
        "wafers": g.size(),
        "defects": g["_d"].sum().astype(int),
    })
    top = d[d["_d"]].groupby("lot_id")["pattern"].agg(lambda s: Counter(s).most_common(1)[0])
    out["top_pattern"] = top.map(lambda x: x[0])
    out["top_pattern_count"] = top.map(lambda x: int(x[1]))
    return out.sort_values("start").reset_index()


# --------------------------------------------------------------------------- overview
def line_overview(stream_df: pd.DataFrame, *, as_of, window_hours: float = 24) -> dict:
    """Whole-line snapshot: totals, every tool side by side, alerts, latest lot."""
    past = stream_as_of(stream_df, as_of)
    if past.empty:
        return {"as_of": pd.Timestamp(as_of).isoformat(), "wafers": 0, "note": "No wafers on the line yet."}
    if not has_predictions(past):
        return {"as_of": pd.Timestamp(as_of).isoformat(), "wafers": int(len(past)),
                "alerts_status": "unavailable_no_predictions",
                "note": "The stream has no predictions yet, so defect rates and alerts are unavailable."}
    t = pd.to_datetime(past["timestamp"])
    recent = past[t > pd.Timestamp(as_of) - pd.Timedelta(hours=window_hours)]
    alerts = check_alert(past)
    alert_tools = {a["tool_id"] for a in alerts}
    tools = []
    for tool, g in past.groupby("tool_id", sort=True):
        d = g[_defect(g)]
        r = recent[(recent["tool_id"] == tool) & _defect(recent)]
        tools.append({
            "tool_id": tool, "wafers": int(len(g)), "lots": int(g["lot_id"].nunique()),
            "defects": int(len(d)), "defect_rate": _round(len(d) / len(g)),
            f"defects_last_{int(window_hours)}h": int(len(r)),
            "top_pattern": Counter(d["pattern"]).most_common(1)[0][0] if len(d) else None,
            "pattern_counts": dict(Counter(d["pattern"]).most_common()),
            "needs_review": int(g["needs_review"].fillna(False).astype(bool).sum()) if "needs_review" in g else 0,
            "in_alert": tool in alert_tools,
            "latest_lot": g.sort_values("timestamp")["lot_id"].iloc[-1],
        })
    lots = _lot_table(past)
    latest = lots.iloc[-1]
    return {
        "as_of": pd.Timestamp(as_of).isoformat(),
        "wafers": int(len(past)), "lots": int(len(lots)),
        "defects": int(_defect(past).sum()), "defect_rate": _round(_defect(past).mean()),
        "needs_review": int(past["needs_review"].fillna(False).astype(bool).sum()) if "needs_review" in past else 0,
        "pattern_counts": dict(Counter(past.loc[_defect(past), "pattern"]).most_common()),
        "tools": tools,
        "ranking_by_defect_rate": [x["tool_id"] for x in sorted(tools, key=lambda x: -x["defect_rate"])],
        "alerts": [{"tool_id": a["tool_id"], "pattern": a["pattern"], "fired_at": a["fired_at"],
                    "fired_lot_id": a["fired_lot_id"], "count_in_window": a["count_in_window"]} for a in alerts],
        "latest_lot": {"lot_id": latest["lot_id"], "tool_id": latest["tool_id"],
                       "wafers_so_far": int(latest["wafers"]), "started": latest["start"].isoformat()},
        "data_note": "Lots, tools and times are simulated; wafer maps are real WM-811K maps.",
    }


# --------------------------------------------------------------------------- trend
def defect_trend(stream_df: pd.DataFrame, *, as_of, tool_id: str | None = None,
                 pattern: str | None = None, onset_rate: float = 0.3) -> dict:
    """Lot-by-lot defect counts in time order, when a pattern first appeared, and the direction.

    `onset` is the first lot at or above `onset_rate` that follows a lot below it: the point a
    tool started producing the pattern in bulk. `direction` compares the last two lots with
    the two before them.
    """
    past = stream_as_of(stream_df, as_of)
    if tool_id:
        past = past[past["tool_id"] == tool_id]
    if past.empty or not has_predictions(past):
        return {"as_of": pd.Timestamp(as_of).isoformat(), "filters": {"tool_id": tool_id, "pattern": pattern},
                "lots": [], "note": "No scored wafers match these filters up to as_of."}
    hit = (past["pattern"] == pattern) if pattern else _defect(past)
    d = past.assign(_hit=hit, _t=pd.to_datetime(past["timestamp"]))
    g = d.groupby("lot_id", sort=False)
    lots = pd.DataFrame({"tool_id": g["tool_id"].first(), "start": g["_t"].min(),
                         "wafers": g.size(), "hits": g["_hit"].sum().astype(int)}).sort_values("start")
    lots["rate"] = lots["hits"] / lots["wafers"]
    rows = [{"lot_id": lot, "tool_id": r.tool_id, "start": r.start.isoformat(), "wafers": int(r.wafers),
             "matching": int(r.hits), "rate": _round(r.rate)} for lot, r in lots.iterrows()]
    first = d[d["_hit"]].sort_values("_t").head(1)
    onset = None
    prev_low = True
    for row in rows:
        if row["rate"] >= onset_rate and prev_low:
            onset = row
        prev_low = row["rate"] < onset_rate
        if onset:
            break
    rates = [r["rate"] for r in rows]
    if len(rates) >= 4:
        a, b = np.mean(rates[-4:-2]), np.mean(rates[-2:])
        direction = "rising" if b - a > 0.1 else "falling" if a - b > 0.1 else "flat"
    else:
        direction = "not enough lots to tell"
    return {
        "as_of": pd.Timestamp(as_of).isoformat(),
        "filters": {"tool_id": tool_id, "pattern": pattern or "any defect"},
        "lots": rows,
        "first_seen": None if first.empty else {
            "timestamp": first["_t"].iloc[0].isoformat(), "lot_id": first["lot_id"].iloc[0],
            "tool_id": first["tool_id"].iloc[0], "uid": int(first["uid"].iloc[0])},
        "onset_lot": onset,
        "onset_rule": f"first lot with >= {int(onset_rate * 100)}% matching wafers after a lot below that",
        "latest_lot_rate": rows[-1]["rate"] if rows else None,
        "direction": direction,
        "direction_rule": "mean rate of the last 2 lots vs the 2 before; >10 points = rising/falling",
    }


# --------------------------------------------------------------------------- wafer search
WAFER_FIELDS = ["uid", "lot_id", "wafer_in_lot", "tool_id", "timestamp", "pattern", "confidence",
                "needs_review", "review_reasons"]


def find_wafers(stream_df: pd.DataFrame, *, as_of, lot_id: str | None = None, tool_id: str | None = None,
                pattern: str | None = None, defects_only: bool = False, needs_review: bool | None = None,
                max_confidence: float | None = None, wafer_in_lot: int | None = None,
                uid: int | None = None, limit: int = 15) -> dict:
    """Wafers on the line up to as_of matching every given filter, newest first."""
    past = stream_as_of(stream_df, as_of)
    m = pd.Series(True, index=past.index)
    if lot_id:
        m &= past["lot_id"].astype(str).str.upper() == str(lot_id).upper()
    if tool_id:
        m &= past["tool_id"].astype(str).str.upper() == str(tool_id).upper()
    if pattern:
        m &= past["pattern"].astype(str).str.lower() == str(pattern).lower()
    if defects_only:
        m &= _defect(past)
    if needs_review is not None and "needs_review" in past:
        m &= past["needs_review"].fillna(False).astype(bool) == bool(needs_review)
    if max_confidence is not None and "confidence" in past:
        m &= past["confidence"] <= float(max_confidence)
    if wafer_in_lot is not None:
        m &= past["wafer_in_lot"] == int(wafer_in_lot)
    if uid is not None:
        m &= past["uid"] == int(uid)
    hits = past[m].sort_values("timestamp", ascending=False)
    cols = [c for c in WAFER_FIELDS if c in hits.columns]
    rows = []
    for _, r in hits.head(max(1, int(limit))).iterrows():
        row = {c: r[c] for c in cols}
        row["timestamp"] = pd.Timestamp(row["timestamp"]).isoformat()
        for k in ("uid", "wafer_in_lot"):
            if k in row:
                row[k] = int(row[k])
        if "confidence" in row:
            row["confidence"] = _round(row["confidence"])
        if "needs_review" in row:
            row["needs_review"] = bool(row["needs_review"])
        if "review_reasons" in row:
            row["review_reasons"] = [x for x in str(row["review_reasons"] or "").split(";") if x and x != "nan"]
        rows.append(row)
    return {"as_of": pd.Timestamp(as_of).isoformat(), "matches": int(len(hits)), "returned": len(rows),
            "truncated": len(hits) > len(rows), "wafers": rows,
            "pattern_counts": dict(Counter(hits["pattern"]).most_common()) if len(hits) else {}}


# --------------------------------------------------------------------------- decisions
def decisions_view(decisions: list[dict], stream_df: pd.DataFrame, *, as_of) -> dict:
    """Engineer decisions up to as_of, and the lots each hold covers.

    A hold on an alert covers the lots in the alert window plus every later lot on that tool,
    matching the dashboard's Lots page.
    """
    t_as_of = pd.Timestamp(as_of)
    seen = [d for d in decisions if pd.Timestamp(d.get("as_of", as_of)) <= t_as_of]
    past = stream_as_of(stream_df, as_of)
    lots = _lot_table(past) if len(past) and has_predictions(past) else pd.DataFrame()
    alerts = check_alert(past) if len(past) and has_predictions(past) else []
    held = []
    for d in seen:
        if d.get("action") != "hold" or lots.empty:
            continue
        window = next((a["lot_ids"] for a in alerts
                       if a["tool_id"] == d["tool_id"] and a["fired_at"] == d.get("fired_at")), [])
        later = lots[(lots["tool_id"] == d["tool_id"]) & (lots["start"] >= pd.Timestamp(d["fired_at"]))]
        held.extend(list(window) + later["lot_id"].tolist())
    return {"as_of": t_as_of.isoformat(), "n_decisions": len(seen),
            "decisions": [{k: d.get(k) for k in ("action", "tool_id", "pattern", "fired_at", "as_of", "by", "note")}
                          for d in seen],
            "held_lots": sorted(set(held)),
            "note": None if seen else "No engineer decisions recorded yet. Alerts stay open until someone acts."}


# --------------------------------------------------------------------------- patterns
def describe_pattern(pattern: str) -> dict:
    """What a pattern looks like, what it is confused with, and how common it is in WM-811K."""
    match = {c.lower(): c for c in CLASSES}.get(str(pattern).strip().lower())
    if match is None:
        raise ValueError(f"unknown pattern {pattern!r}; known: {CLASSES}")
    confusable = sorted({b if a == match else a for a, b in CONFUSABLE_PAIRS if match in (a, b)})
    total = sum(DATASET_COUNTS.values())
    defects = total - DATASET_COUNTS["none"]
    return {"pattern": match, "looks_like": PATTERN_LOOKS[match], "often_confused_with": confusable,
            "wm811k_labelled_count": DATASET_COUNTS[match],
            "share_of_labelled_maps": _round(DATASET_COUNTS[match] / total, 4),
            "share_of_defect_maps": None if match == "none" else _round(DATASET_COUNTS[match] / defects, 4)}


# --------------------------------------------------------------------------- explanations
_DIRS = ["right", "lower right", "bottom", "lower left", "left", "upper left", "top", "upper right"]


def evidence_location(wafer_map: np.ndarray, heatmap: np.ndarray) -> dict:
    """Where on the wafer the model's attention sits, in words an engineer uses.

    zone: heat-weighted mean distance of the failing dies from the wafer centre
    (0 = centre, 1 = edge), so a full ring reads as "edge", not as its centre of mass.
    direction: where the heat is concentrated (row 0 is the top of the map), or
    "all around" when it is spread evenly, as on a ring.
    """
    m = np.asarray(wafer_map)
    h = np.clip(np.asarray(heatmap, dtype=np.float64), 0, None)
    dies = m != NO_DIE
    fail = m == FAIL
    rr, cc = np.nonzero(dies)
    if not len(rr):
        return {"zone": "unknown"}
    cy, cx = rr.mean(), cc.mean()
    radius = max(1.0, math.sqrt(dies.sum() / math.pi))
    w = h * fail
    if w.sum() <= 0:
        w = fail.astype(np.float64)
    if w.sum() <= 0:
        return {"zone": "no failing dies"}
    fy, fx = np.nonzero(w)
    wt = w[fy, fx] / w[fy, fx].sum()
    dy, dx = (fy - cy) / radius, (fx - cx) / radius
    r_mean = float(np.clip((np.hypot(dy, dx) * wt).sum(), 0, 1))
    zone = "centre" if r_mean < 0.38 else "mid-radius" if r_mean < 0.75 else "edge"   # uniform disk = 0.67
    my, mx = float((dy * wt).sum()), float((dx * wt).sum())
    offset = math.hypot(my, mx)
    if zone == "centre":
        direction = "centre"
    elif offset < 0.35 * r_mean:
        direction = "all around"
    else:
        direction = _DIRS[int(((math.degrees(math.atan2(my, mx)) % 360 + 22.5) % 360) // 45)]
    # how concentrated the attention is: share of heat on the hottest 10% of failing dies
    v = np.sort(h[fail])[::-1]
    top = v[: max(1, len(v) // 10)].sum() / max(v.sum(), 1e-9)
    return {"zone": zone, "direction": direction, "mean_radius_fraction": _round(r_mean, 2),
            "attention_concentration": _round(top, 2),
            "fail_dies": int(fail.sum()), "dies": int(dies.sum()), "fail_share": _round(fail.sum() / dies.sum())}


# --------------------------------------------------------------------------- model quality
def model_performance(metrics: dict, pattern: str | None = None) -> dict:
    """How well the line model does, from its validation metrics.json, per pattern if asked.

    `most_confused_with` comes from the confusion matrix row of the true pattern.
    """
    per = {r["class"]: r for r in metrics["per_class"]}
    conf = np.asarray(metrics.get("confusion") or [], dtype=float)
    review = metrics.get("review_rate_per_class") or {}

    def one(c: str) -> dict:
        r = per[c]
        out = {"pattern": c, "precision": _round(r["precision"]), "recall": _round(r["recall"]),
               "f1": _round(r["f1"]), "validation_wafers": int(r["support"]),
               "sent_to_review": _round(review[c]) if c in review else None}
        if conf.size:
            i = CLASSES.index(c)
            row = conf[i].copy()
            row[i] = 0
            if row.sum() > 0:
                j = int(row.argmax())
                out["most_confused_with"] = {"pattern": CLASSES[j], "share_of_this_pattern": _round(row[j] / conf[i].sum())}
        return out

    ci = (metrics.get("ci") or {}).get("macro_f1") or {}
    summary = {
        "model": str(metrics.get("model", "")).split("/")[-1], "split": metrics.get("split"),
        "validation_wafers": metrics.get("n"),
        "macro_f1": _round(metrics["macro_f1"]),
        "macro_f1_95ci": [_round(ci["lo"]), _round(ci["hi"])] if ci else None,
        "accuracy": _round(metrics["accuracy"]),
        "defect_recall": _round(metrics["defect_recall"]), "defect_precision": _round(metrics["defect_precision"]),
        "calibration_error_ece": _round(metrics["ece"], 4), "review_rate": _round(metrics["review_rate"], 4),
        "evaluation_note": ("Validation lots, grouped by lot so no lot is in both training and evaluation. "
                            "Calibration was fitted on this set, so the numbers are slightly optimistic. "
                            "The held-out test set has not been scored yet."),
        "why_macro_f1": "85% of wafers have no pattern, so accuracy looks high; macro-F1 weighs all 9 patterns equally.",
    }
    if pattern:
        match = {c.lower(): c for c in CLASSES}.get(str(pattern).strip().lower())
        if match is None or match not in per:
            raise ValueError(f"unknown pattern {pattern!r}; known: {CLASSES}")
        summary["pattern"] = one(match)
    else:
        summary["per_pattern"] = [one(c) for c in CLASSES if c in per]
        ranked = sorted(summary["per_pattern"], key=lambda x: x["f1"])
        summary["weakest_patterns"] = [x["pattern"] for x in ranked[:2]]
        summary["strongest_patterns"] = [x["pattern"] for x in ranked[-2:][::-1]]
    return summary

"""Lot / tool history for the agent, scoped to a replay clock (as_of)."""
from __future__ import annotations

from collections import Counter

import pandas as pd

from wafer.factory.alerts import check_alert
from wafer.factory.impact import impact_from_alert

EVENT_COLS = ("lot_id", "tool_id", "uid", "timestamp", "pattern", "confidence",
              "needs_review", "review_reasons", "model_version", "classifier")

NO_ALERT_NOTE = (
    "No prediction-based alert has fired yet under the configured rule on the "
    "predictions available as of as_of. That establishes only that the alert "
    "threshold was not met — not that there is no exposure, and not that no "
    "action is warranted."
)


def stream_as_of(stream_df: pd.DataFrame, as_of) -> pd.DataFrame:
    """Rows with timestamp <= as_of (inclusive). Never includes future events."""
    if "timestamp" not in stream_df.columns:
        raise ValueError("stream_df is missing column 'timestamp'")
    t = pd.to_datetime(stream_df["timestamp"])
    return stream_df.loc[t <= pd.Timestamp(as_of)].copy()


def has_predictions(stream_df: pd.DataFrame) -> bool:
    return "pattern" in stream_df.columns and stream_df["pattern"].notna().any()


def _row_event(row, cols) -> dict:
    ev = {c: row[c] for c in cols if c in row.index and pd.notna(row[c])}
    if "timestamp" in ev:
        ev["timestamp"] = pd.Timestamp(ev["timestamp"]).isoformat()
    if "uid" in ev:
        ev["uid"] = int(ev["uid"])
    if "confidence" in ev:
        ev["confidence"] = float(ev["confidence"])
    if "needs_review" in ev:
        ev["needs_review"] = bool(ev["needs_review"])
    return ev


def _pattern_counts(series) -> dict:
    return {str(k): int(v) for k, v in Counter(series.dropna().astype(str)).items()}


def _lot_summaries(events_df: pd.DataFrame) -> list[dict]:
    if events_df.empty:
        return []
    rows = []
    gcols = ["lot_id"]
    if "tool_id" in events_df.columns:
        gcols = ["lot_id", "tool_id"]
    for keys, g in events_df.sort_values("timestamp", kind="stable").groupby(gcols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        lot_id = str(keys[0])
        tool = str(keys[1]) if len(keys) > 1 else (str(g["tool_id"].iloc[0]) if "tool_id" in g else None)
        summary = {
            "lot_id": lot_id,
            "tool_id": tool,
            "n_wafers": int(len(g)),
            "first_timestamp": pd.Timestamp(g["timestamp"].iloc[0]).isoformat(),
            "last_timestamp": pd.Timestamp(g["timestamp"].iloc[-1]).isoformat(),
        }
        if "pattern" in g.columns:
            summary["pattern_counts"] = _pattern_counts(g["pattern"])
        if "needs_review" in g.columns:
            summary["n_needs_review"] = int(g["needs_review"].fillna(False).astype(bool).sum())
        rows.append(summary)
    return rows


def _review_candidates(events_df: pd.DataFrame, limit: int = 10) -> list[dict]:
    if "needs_review" not in events_df.columns or events_df.empty:
        return []
    rev = events_df[events_df["needs_review"].fillna(False).astype(bool)]
    rev = rev.sort_values("timestamp", kind="stable")
    cols = [c for c in EVENT_COLS if c in rev.columns]
    return [_row_event(row, cols) for _, row in rev.tail(limit).iterrows()]


def _compact_alert(alert: dict) -> dict:
    uids = list(alert.get("uids") or [])
    out = {k: v for k, v in alert.items() if k != "uids"}
    out["uids_count"] = len(uids)
    out["uids_sample"] = uids[:5]
    return out


def check_lot_history(stream_df: pd.DataFrame, *, as_of, lot_id: str | None = None,
                      tool_id: str | None = None, pattern: str | None = None,
                      limit: int = 8, window_hours: float = 24, threshold: int = 3,
                      expected_classifier: str | None = None,
                      review_limit: int = 10) -> dict:
    """Compact history + alerts visible at `as_of` (for the LLM).

    Computes alerts, lot summaries, counts, and review candidates on the **full**
    filtered history before taking an event sample. The payload returned to the model
    is compact; `full_events` is included for local callers (the agent tool strips it
    before sending to the LLM and keeps it on the session).
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")
    past = stream_as_of(stream_df, as_of)
    filters = {"lot_id": lot_id, "tool_id": tool_id, "pattern": pattern}

    classifier_note = None
    if expected_classifier and "classifier" in past.columns and len(past):
        seen = sorted({str(c) for c in past["classifier"].dropna().unique()})
        if seen and seen != [expected_classifier]:
            classifier_note = (f"stream predictions are from classifier(s) {seen}; "
                               f"selected backend is {expected_classifier!r}")

    # --- alerts: full history up to as_of; never filter by lot_id
    if not has_predictions(past):
        alerts: list = []
        alerts_status = "unavailable_no_predictions"
        alerts_note = ("No classifier predictions on the stream. Score it with the CNN "
                       "(e.g. score_stream / make_stream.py) before asking for alerts. "
                       "True labels are never used as predictions.")
    else:
        alert_df = past
        if tool_id is not None:
            alert_df = alert_df[alert_df["tool_id"] == tool_id]
        if pattern is not None:
            alert_df = alert_df[alert_df["pattern"] == pattern]
        alerts = check_alert(alert_df, window_hours=window_hours, threshold=threshold)
        alerts_status = "ok"
        alerts_note = None if alerts else NO_ALERT_NOTE

    # --- full filtered event set (then sample)
    events_df = past
    if lot_id is not None:
        events_df = events_df[events_df["lot_id"] == lot_id]
    if tool_id is not None:
        events_df = events_df[events_df["tool_id"] == tool_id]
    if pattern is not None and "pattern" in events_df.columns:
        events_df = events_df[events_df["pattern"] == pattern]
    events_df = events_df.sort_values("timestamp", kind="stable")
    full_event_count = int(len(events_df))

    lot_summaries = _lot_summaries(events_df)
    review_candidates = _review_candidates(events_df, limit=review_limit)

    counts = {
        "events_matching_filters": full_event_count,
        "lots": int(events_df["lot_id"].nunique()) if "lot_id" in events_df.columns and full_event_count else 0,
        "tools": int(events_df["tool_id"].nunique()) if "tool_id" in events_df.columns and full_event_count else 0,
    }
    if "pattern" in events_df.columns and full_event_count:
        counts["pattern_counts"] = _pattern_counts(events_df["pattern"])
    if "needs_review" in events_df.columns and full_event_count:
        counts["n_needs_review"] = int(events_df["needs_review"].fillna(False).astype(bool).sum())

    cols = [c for c in EVENT_COLS if c in events_df.columns]
    full_events = [_row_event(row, cols) for _, row in events_df.iterrows()]
    truncated = full_event_count > limit
    sample = full_events[-limit:] if truncated else full_events
    counts["events_in_sample"] = len(sample)

    return {
        "as_of": pd.Timestamp(as_of).isoformat(),
        "filters": filters,
        "counts": counts,
        "lot_summaries": lot_summaries,
        "alerts": [_compact_alert(a) for a in alerts],
        "alerts_status": alerts_status,
        "alerts_note": alerts_note,
        "review_candidates": review_candidates,
        "event_sample": sample,
        "event_count": len(sample),  # back-compat: sample size
        "events_matching_filters": full_event_count,
        "events_truncated": truncated,
        "events": sample,  # back-compat alias for sample
        "full_events": full_events,  # local only — strip before LLM
        "full_events_available_locally": True,
        "classifier_note": classifier_note,
        "data_source": "simulated_factory_stream",
        "data_note": ("Lot IDs, tool IDs and equipment history are simulated for the demo. "
                      "Wafer maps are real WM-811K maps."),
    }


def history_for_llm(payload: dict) -> dict:
    """Drop locally retained full event lists before sending to the model."""
    return {k: v for k, v in payload.items() if k != "full_events"}


def value_at_risk_for_alert(stream_df: pd.DataFrame, alert: dict, *, as_of,
                            wafer_value: float | None = None,
                            wafers_per_lot: int | None = None) -> dict:
    """Impact from simulation evidence at `as_of` (subsequent lots observed since the alert)."""
    past = stream_as_of(stream_df, as_of)
    return impact_from_alert(past, alert, wafer_value=wafer_value, wafers_per_lot=wafers_per_lot,
                             as_of=pd.Timestamp(as_of).isoformat())

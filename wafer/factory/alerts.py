"""Excursion alert rule (the agent's `check_alert` tool).

Fires when `threshold` or more wafers from the same tool show the same defect pattern within
`window_hours`. Per docs/HANDOFF.md the rule reads the model's prediction (`pattern`, from
wafer.infer.classify), never the true `label`: the demo must react to what the fab would see.
"""
from __future__ import annotations

import pandas as pd

IGNORED_PATTERNS = ("none",)


def check_alert(stream_df: pd.DataFrame, window_hours: float = 24, threshold: int = 3,
                pattern_col: str = "pattern", ignore: tuple[str, ...] = IGNORED_PATTERNS) -> list[dict]:
    """One alert per (tool, pattern): the first moment the rule fires, oldest first.

    stream_df needs tool_id, timestamp, lot_id, uid and `pattern_col`. Pass the wafers seen so
    far to evaluate a live stream. Returns plain Python types (JSON-serialisable).
    """
    if threshold < 1 or window_hours <= 0:
        raise ValueError("threshold must be >= 1 and window_hours > 0")
    missing = {"tool_id", "timestamp", "lot_id", "uid", pattern_col} - set(stream_df.columns)
    if missing:
        raise ValueError(f"stream_df is missing columns {sorted(missing)}")

    df = stream_df[~stream_df[pattern_col].isin(ignore)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    window = pd.Timedelta(hours=window_hours)
    alerts = []
    for (tool, pattern), g in df.sort_values("timestamp", kind="stable").groupby(["tool_id", pattern_col], sort=False):
        t = g["timestamp"].tolist()
        start = 0
        for end in range(len(t)):
            while t[end] - t[start] >= window:       # keep events in the half-open window (t_end - w, t_end]
                start += 1
            if end - start + 1 >= threshold:
                hit = g.iloc[start:end + 1]
                alerts.append({
                    "tool_id": str(tool), "pattern": str(pattern),
                    "count_in_window": int(len(hit)), "threshold": int(threshold),
                    "window_hours": float(window_hours),
                    "first_seen": hit["timestamp"].iloc[0].isoformat(),
                    "fired_at": hit["timestamp"].iloc[-1].isoformat(),
                    "fired_lot_id": str(hit["lot_id"].iloc[-1]),
                    "lot_ids": sorted(map(str, hit["lot_id"].unique())),
                    "uids": [int(u) for u in hit["uid"]],
                })
                break
    return sorted(alerts, key=lambda a: a["fired_at"])


def lots_caught_early(stream_df: pd.DataFrame, alert: dict) -> int:
    """Lots that would have started on the alert's tool after it fired: the lots a tool hold saves."""
    df = stream_df[stream_df["tool_id"] == alert["tool_id"]]
    starts = pd.to_datetime(df.groupby("lot_id")["timestamp"].min())
    return int((starts > pd.Timestamp(alert["fired_at"])).sum())

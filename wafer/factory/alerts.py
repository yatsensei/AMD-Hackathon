"""Excursion alert rule (the agent's `check_alert` tool).

Fires when `threshold` or more wafers from the same tool show the same defect pattern within
`window_hours`. Per docs/HANDOFF.md the rule reads the model's prediction (`pattern`, from
wafer.infer.classify), never the true `label`: the demo must react to what the fab would see.

By default a tool raises one alert: patterns that fire later on a tool already in alert are
listed under that alert's `also_seen`. On the scored val stream the CNN reads 5 of the 275
ETCH-04 Edge-Ring wafers as Edge-Loc, which alone trips the 3-in-24h rule; reported as its
own alert it looks like a second, lithography-caused fault on the same tool.
"""
from __future__ import annotations

import pandas as pd

IGNORED_PATTERNS = ("none",)


def check_alert(stream_df: pd.DataFrame, window_hours: float = 24, threshold: int = 3,
                pattern_col: str = "pattern", ignore: tuple[str, ...] = IGNORED_PATTERNS,
                one_per_tool: bool = True) -> list[dict]:
    """The first moment the rule fires per (tool, pattern), oldest first.

    one_per_tool=True keeps the first alert per tool and moves later patterns on that tool
    into its `also_seen` list ({pattern, fired_at, fired_lot_id, count_in_window}).

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
    alerts.sort(key=lambda a: a["fired_at"])
    if not one_per_tool:
        return alerts
    primary: dict[str, dict] = {}
    for a in alerts:
        if a["tool_id"] in primary:
            primary[a["tool_id"]]["also_seen"].append(
                {k: a[k] for k in ("pattern", "fired_at", "fired_lot_id", "count_in_window")})
        else:
            primary[a["tool_id"]] = {**a, "also_seen": []}
    return list(primary.values())


def lots_caught_early(stream_df: pd.DataFrame, alert: dict) -> int:
    """Count subsequent lots on the alert's tool whose lot start is after `fired_at`.

    This is an observed-count helper for the stream slice you pass in (typically
    timestamp <= as_of). It is not proof that a hold happened or that money was saved.
    """
    df = stream_df[stream_df["tool_id"] == alert["tool_id"]]
    starts = pd.to_datetime(df.groupby("lot_id")["timestamp"].min())
    return int((starts > pd.Timestamp(alert["fired_at"])).sum())

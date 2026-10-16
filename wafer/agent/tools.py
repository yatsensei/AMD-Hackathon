"""OpenAI tool schemas and execution for the wafer agent."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from wafer.agent.session import AgentContext
from wafer.classifiers.base import ClassifierNotAvailableError
from wafer.factory import (
    check_lot_history,
    estimate_impact,
    lookup_root_cause,
    value_at_risk_for_alert,
)
from wafer.factory.history import has_predictions, history_for_llm, stream_as_of
from wafer.factory.alerts import check_alert

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "classify_wafer",
            "description": ("Classify one wafer map by uid using the selected classifier backend "
                            "(CNN by default). Returns pattern, confidence, needs_review, "
                            "model_version, classifier identity."),
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {"type": "integer", "description": "Wafer uid from the stream / dataset"},
                },
                "required": ["uid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_root_cause",
            "description": ("Look up possible root cause, machine family and recommended action "
                            "for a defect pattern. Preserves source and source_warnings. "
                            "Call in parallel with estimate_impact after history when possible."),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Defect class name, e.g. Edge-Ring"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_lot_history",
            "description": (
                "Compact simulated history up to as_of: counts, per-lot summaries, alerts, "
                "review candidates, and a small event sample. Alerts use full tool history "
                "(lot filters do not hide tool-wide recurrence). Call once per status question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lot_id": {"type": "string", "description": "Optional lot filter for summaries/sample"},
                    "tool_id": {"type": "string", "description": "Optional tool filter"},
                    "pattern": {"type": "string", "description": "Optional predicted pattern filter"},
                    "limit": {"type": "integer", "description": "Max events in sample (default 8)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_impact",
            "description": (
                "Exposure from simulation evidence or a manual hypothetical count. "
                "Reports observed_wafers vs assumed_full_lot_*; never treat unobserved "
                "wafers as processed. Call in parallel with lookup_root_cause after history."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_id": {"type": "string", "description": "Tool to find an alert for (simulation evidence)"},
                    "pattern": {"type": "string", "description": "Optional pattern when resolving an alert"},
                    "lots_caught_early": {
                        "type": "integer",
                        "description": "Manual lot count — future_hypothetical_exposure only",
                    },
                    "wafer_value": {"type": "number", "description": "Optional USD per wafer override"},
                    "wafers_per_lot": {"type": "integer", "description": "Optional wafers per lot override"},
                },
                "required": [],
            },
        },
    },
]


def _ok(payload: dict) -> str:
    return json.dumps({"ok": True, **payload}, default=str)


def _err(message: str, **extra) -> str:
    return json.dumps({"ok": False, "error": message, **extra}, default=str)


def _get_map(ctx: AgentContext, uid: int):
    if uid in ctx.maps:
        return ctx.maps[uid]
    from wafer.data.access import get_wafer_map
    return get_wafer_map(uid)


def _classify_wafer(args: dict, ctx: AgentContext) -> str:
    if "uid" not in args:
        return _err("uid is required")
    uid = int(args["uid"])
    try:
        wafer_map = _get_map(ctx, uid)
    except FileNotFoundError as exc:
        return _err(str(exc))
    except Exception as exc:
        return _err(f"could not load wafer map for uid={uid}: {exc}")
    try:
        result = ctx.classifier.classify(wafer_map)
    except ClassifierNotAvailableError as exc:
        return _err(str(exc), classifier=ctx.classifier_name)
    except Exception as exc:
        return _err(f"classification failed: {exc}", classifier=ctx.classifier_name)
    return _ok({"uid": uid, "result": result})


def _lookup_root_cause(args: dict, ctx: AgentContext) -> str:
    pattern = args.get("pattern")
    if not pattern:
        return _err("pattern is required")
    try:
        return _ok(lookup_root_cause(str(pattern)))
    except ValueError as exc:
        return _err(str(exc))


def _check_lot_history(args: dict, ctx: AgentContext) -> str:
    try:
        result = check_lot_history(
            ctx.stream_df,
            as_of=ctx.as_of,
            lot_id=args.get("lot_id"),
            tool_id=args.get("tool_id"),
            pattern=args.get("pattern"),
            limit=int(args.get("limit", 8)),
            expected_classifier=ctx.classifier_name,
        )
        ctx.last_history_full = result
        return _ok(history_for_llm(result))
    except Exception as exc:
        return _err(f"check_lot_history failed: {exc}")


def _estimate_impact(args: dict, ctx: AgentContext) -> str:
    wafer_value = args.get("wafer_value")
    wafers_per_lot = args.get("wafers_per_lot")
    tool_id = args.get("tool_id")
    pattern = args.get("pattern")
    manual = args.get("lots_caught_early")

    kwargs: dict[str, Any] = {}
    if wafer_value is not None:
        kwargs["wafer_value"] = float(wafer_value)
    if wafers_per_lot is not None:
        kwargs["wafers_per_lot"] = int(wafers_per_lot)

    if tool_id is not None:
        past = stream_as_of(ctx.stream_df, ctx.as_of)
        if not has_predictions(past):
            return _err(
                "Cannot derive lots from simulation: stream has no predictions. "
                "Score the stream first, or pass lots_caught_early (hypothetical).",
                alerts_status="unavailable_no_predictions",
            )
        alert_df = past[past["tool_id"] == tool_id]
        if pattern is not None:
            alert_df = alert_df[alert_df["pattern"] == pattern]
        alerts = check_alert(alert_df)
        if not alerts:
            return _err(
                f"no prediction-based alert found for tool_id={tool_id!r} up to as_of={ctx.as_of}. "
                "That means the alert threshold was not met on available predictions — "
                "not that exposure is zero or that no action is warranted.",
                alerts_status="ok",
                alerts=[],
            )
        alert = next((a for a in alerts if pattern and a["pattern"] == pattern), alerts[0])
        try:
            return _ok(value_at_risk_for_alert(ctx.stream_df, alert, as_of=ctx.as_of, **kwargs))
        except Exception as exc:
            return _err(f"estimate_impact failed: {exc}")

    if manual is not None:
        try:
            out = estimate_impact(int(manual), hypothetical=True,
                                  as_of=pd.Timestamp(ctx.as_of).isoformat(), **kwargs)
            return _ok(out)
        except Exception as exc:
            return _err(f"estimate_impact failed: {exc}")

    return _err("Provide tool_id (preferred, simulation evidence) or lots_caught_early "
                "(explicitly hypothetical).")


_HANDLERS = {
    "classify_wafer": _classify_wafer,
    "lookup_root_cause": _lookup_root_cause,
    "check_lot_history": _check_lot_history,
    "estimate_impact": _estimate_impact,
}


def execute_tool(name: str, arguments: dict | str, ctx: AgentContext) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return _err(f"invalid tool arguments JSON: {exc}")
    handler = _HANDLERS.get(name)
    if handler is None:
        return _err(f"unknown tool {name!r}", known=sorted(_HANDLERS))
    try:
        return handler(arguments or {}, ctx)
    except Exception as exc:
        return _err(f"tool {name} failed: {exc}")

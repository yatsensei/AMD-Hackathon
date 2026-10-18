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
from wafer.factory import insights

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
    {
        "type": "function",
        "function": {
            "name": "line_overview",
            "description": ("Whole-line snapshot up to as_of: totals, every tool side by side (wafers, "
                            "defect rate, last-24h defects, top pattern, pattern counts, review count, "
                            "alert flag), tools ranked by defect rate, alerts, latest lot. Use for "
                            "summaries, comparisons and 'which tool/pattern is worst' questions."),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "defect_trend",
            "description": ("Lot-by-lot defect counts in time order for a tool and/or pattern, when the "
                            "pattern was first seen, the onset lot, the latest lot's rate and whether it is "
                            "rising, falling or flat. Use for 'when did it start', 'is it getting worse'."),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_id": {"type": "string", "description": "Optional tool, e.g. ETCH-04"},
                    "pattern": {"type": "string", "description": "Optional pattern; default = any defect"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_wafers",
            "description": ("Search wafers on the line up to as_of, newest first. Combine any filters: "
                            "lot, tool, pattern, defects only, needs review, max confidence, wafer position "
                            "in the lot, uid. Returns uid, lot, position, tool, time, pattern, confidence, "
                            "review reasons, and pattern counts over all matches. Use to resolve "
                            "'wafer 6 of LOT-014' to a uid, or to list wafers."),
            "parameters": {
                "type": "object",
                "properties": {
                    "lot_id": {"type": "string"},
                    "tool_id": {"type": "string"},
                    "pattern": {"type": "string"},
                    "defects_only": {"type": "boolean"},
                    "needs_review": {"type": "boolean"},
                    "max_confidence": {"type": "number", "description": "e.g. 0.8 for uncertain wafers"},
                    "wafer_in_lot": {"type": "integer", "description": "Position 1-25 in the lot"},
                    "uid": {"type": "integer"},
                    "limit": {"type": "integer", "description": "Max wafers returned (default 15)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_wafer",
            "description": ("Why the model called a wafer what it did: line prediction from the stream, "
                            "live classification with the top-3 alternatives, where on the wafer the "
                            "evidence sits (centre / mid-radius / edge and direction) and how many dies "
                            "fail. Use for 'why', 'what else could it be', 'where is the defect'."),
            "parameters": {
                "type": "object",
                "properties": {"uid": {"type": "integer", "description": "Wafer uid (use find_wafers first if you only know lot and position)"}},
                "required": ["uid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_pattern",
            "description": ("What a defect pattern looks like, which patterns it is confused with, how "
                            "common it is in WM-811K, and its possible root cause with source. Use for "
                            "'what is a Donut', 'Edge-Loc vs Edge-Ring', general pattern questions."),
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "model_performance",
            "description": ("How reliable the line model is, from its validation results: macro-F1 with "
                            "95% CI, defect recall and precision, calibration, review rate, and per-pattern "
                            "precision / recall / F1 / most-confused-with. Use for accuracy and trust questions."),
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string", "description": "Optional single pattern"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_decisions",
            "description": ("Engineer decisions on alerts up to as_of (hold, ticket, dismiss; who and when) "
                            "and the lots currently held. Use for 'what have we done', 'which lots are held'."),
            "parameters": {"type": "object", "properties": {}, "required": []},
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


def _line_overview(args: dict, ctx: AgentContext) -> str:
    return _ok(insights.line_overview(ctx.stream_df, as_of=ctx.as_of))


def _defect_trend(args: dict, ctx: AgentContext) -> str:
    return _ok(insights.defect_trend(ctx.stream_df, as_of=ctx.as_of,
                                     tool_id=args.get("tool_id"), pattern=args.get("pattern")))


def _find_wafers(args: dict, ctx: AgentContext) -> str:
    keys = ("lot_id", "tool_id", "pattern", "defects_only", "needs_review", "max_confidence",
            "wafer_in_lot", "uid")
    kw = {k: args[k] for k in keys if args.get(k) is not None}
    return _ok(insights.find_wafers(ctx.stream_df, as_of=ctx.as_of, limit=int(args.get("limit") or 15), **kw))


def _explain_wafer(args: dict, ctx: AgentContext) -> str:
    if "uid" not in args:
        return _err("uid is required")
    uid = int(args["uid"])
    on_line = insights.find_wafers(ctx.stream_df, as_of=ctx.as_of, uid=uid)["wafers"]
    if not on_line and len(ctx.stream_df) and uid in set(ctx.stream_df["uid"].astype(int)):
        return _err(f"wafer {uid} has not reached the line yet at as_of={ctx.as_of}")
    try:
        wafer_map = _get_map(ctx, uid)
    except Exception as exc:
        return _err(f"could not load wafer map for uid={uid}: {exc}")
    out: dict[str, Any] = {"uid": uid, "line_prediction": on_line[0] if on_line else None}
    try:
        live = ctx.classifier.classify(wafer_map)
        probs = live.get("probs") or {}
        out["live_classification"] = {
            "pattern": live.get("pattern"), "confidence": live.get("confidence"),
            "needs_review": live.get("needs_review"), "review_reasons": live.get("review_reasons"),
            "classifier": live.get("classifier", ctx.classifier_name), "model_version": live.get("model_version"),
            "top3": [{"pattern": k, "prob": round(float(v), 4)} for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:3]],
        }
    except Exception as exc:
        out["live_classification"] = None
        out["classification_error"] = str(exc)
    explain = ctx.explain_fn
    if explain is None and ctx.classifier_name == "cnn":
        from wafer.infer import explain as cnn_explain  # lazy: loads the model on first use
        explain = cnn_explain
    if explain is None:
        out["evidence"] = None
        out["evidence_note"] = "Attention maps come from the CNN only; not available for this backend."
    else:
        try:
            e = explain(wafer_map)
            out["evidence"] = insights.evidence_location(wafer_map, e["heatmap"])
            out["evidence"]["box_rows_cols"] = e.get("bbox")
            out["evidence_note"] = ("Grad-CAM attention from one CNN member: shows which failing dies drove "
                                    "the call. It explains the classification; it is not a defect detector.")
        except Exception as exc:
            out["evidence"] = None
            out["evidence_note"] = f"explanation failed: {exc}"
    return _ok(out)


def _describe_pattern(args: dict, ctx: AgentContext) -> str:
    try:
        out = insights.describe_pattern(str(args.get("pattern", "")))
    except ValueError as exc:
        return _err(str(exc))
    rc = lookup_root_cause(out["pattern"])
    out["possible_root_cause"] = {k: rc.get(k) for k in ("cause", "machine", "recommended_action", "source",
                                                         "source_warnings", "is_possibility")}
    return _ok(out)


def _load_metrics(ctx: AgentContext) -> dict:
    if ctx.metrics is not None:
        return ctx.metrics
    import os
    from pathlib import Path
    from wafer.env import REPO_ROOT
    name = Path(os.environ.get("WAFER_MODEL_DIR") or "v0-ens3").name
    f = REPO_ROOT / "docs" / "results" / f"{name}_val" / "metrics.json"
    if not f.exists():
        f = REPO_ROOT / "docs" / "results" / "v0-ens3_val" / "metrics.json"
    ctx.metrics = json.loads(f.read_text())
    return ctx.metrics


def _model_performance(args: dict, ctx: AgentContext) -> str:
    if ctx.classifier_name != "cnn":
        note = ("These are the CNN's validation results. The current line model is "
                f"{ctx.classifier_name!r}, which has no comparable held-out evaluation.")
    else:
        note = None
    try:
        out = insights.model_performance(_load_metrics(ctx), args.get("pattern"))
    except (ValueError, FileNotFoundError) as exc:
        return _err(str(exc))
    if note:
        out["backend_note"] = note
    return _ok(out)


def _get_decisions(args: dict, ctx: AgentContext) -> str:
    if ctx.decisions is None:
        return _err("Decision log is not connected in this session (only the dashboard records decisions).")
    return _ok(insights.decisions_view(ctx.decisions(), ctx.stream_df, as_of=ctx.as_of))


_HANDLERS = {
    "classify_wafer": _classify_wafer,
    "lookup_root_cause": _lookup_root_cause,
    "check_lot_history": _check_lot_history,
    "estimate_impact": _estimate_impact,
    "line_overview": _line_overview,
    "defect_trend": _defect_trend,
    "find_wafers": _find_wafers,
    "explain_wafer": _explain_wafer,
    "describe_pattern": _describe_pattern,
    "model_performance": _model_performance,
    "get_decisions": _get_decisions,
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

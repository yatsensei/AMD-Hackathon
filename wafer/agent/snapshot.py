"""A short, factual line snapshot sent with every question.

It tells the model the valid tool and lot ids, the replay clock, the alerts and what the
engineer is looking at, so simple questions need no tool call and ids are never guessed.
Numbers come from the same functions the tools use.
"""
from __future__ import annotations

import pandas as pd

from wafer.agent.session import AgentContext
from wafer.factory import insights
from wafer.factory.history import has_predictions, stream_as_of


def _hm(ts) -> str:
    return pd.Timestamp(ts).strftime("%a %d %b %H:%M")


def line_snapshot(ctx: AgentContext, ui_context: dict | None = None) -> str:
    past = stream_as_of(ctx.stream_df, ctx.as_of)
    lines = [f"Line time (as_of): {_hm(ctx.as_of)}. You only know wafers up to this time."]
    if past.empty:
        lines.append("No wafers have reached the line yet.")
    elif not has_predictions(past):
        lines.append(f"{len(past)} wafers on the line, but the stream is not scored yet.")
    else:
        o = insights.line_overview(ctx.stream_df, as_of=ctx.as_of)
        lots = sorted(past["lot_id"].astype(str).unique())
        lines.append(f"Tools: {', '.join(t['tool_id'] for t in o['tools'])}. "
                     f"Lots so far: {lots[0]} to {lots[-1]} ({o['lots']} lots, 25 wafers each; "
                     f"latest {o['latest_lot']['lot_id']} on {o['latest_lot']['tool_id']}, "
                     f"{o['latest_lot']['wafers_so_far']} wafers so far).")
        lines.append(f"Wafers inspected: {o['wafers']}; defective {o['defects']} ({o['defect_rate']:.0%}); "
                     f"sent to review {o['needs_review']}.")
        per_tool = "; ".join(f"{t['tool_id']} {t['defect_rate']:.0%} defective"
                             f"{' (top ' + t['top_pattern'] + ')' if t['top_pattern'] else ''}"
                             f"{' ALERT' if t['in_alert'] else ''}" for t in o["tools"])
        lines.append(f"By tool: {per_tool}.")
        if o["alerts"]:
            lines.append("Alerts: " + "; ".join(f"{a['pattern']} on {a['tool_id']}, fired {_hm(a['fired_at'])} "
                                                f"in {a['fired_lot_id']}" for a in o["alerts"]) + ".")
        else:
            lines.append("Alerts: the alert rule has not fired yet.")
    if ctx.decisions is not None:
        d = insights.decisions_view(ctx.decisions(), ctx.stream_df, as_of=ctx.as_of)
        if d["n_decisions"]:
            acts = "; ".join(f"{x['action']} on {x['tool_id']} {x['pattern']} by {x['by']}" for x in d["decisions"][-3:])
            lines.append(f"Engineer decisions: {acts}. Held lots: {', '.join(d['held_lots']) or 'none'}.")
        else:
            lines.append("Engineer decisions: none yet.")
    lines.append(f"Line model: {ctx.classifier_name}.")
    focus = _focus(ui_context)
    if focus:
        lines.append(f"The engineer is currently looking at: {focus}. Resolve 'this', 'it' and 'here' to that.")
    return "\n".join(lines)


def _focus(ui: dict | None) -> str:
    if not ui:
        return ""
    bits = []
    if ui.get("page"):
        bits.append(f"the {ui['page']} page")
    if ui.get("tool_id"):
        bits.append(f"tool {ui['tool_id']}")
    if ui.get("lot_id"):
        bits.append(f"lot {ui['lot_id']}")
    if ui.get("wafer_uid"):
        w = f"wafer uid {int(ui['wafer_uid'])}"
        if ui.get("wafer_label"):
            w += f" ({ui['wafer_label']})"
        bits.append(w)
    return ", ".join(str(b) for b in bits)

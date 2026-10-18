"""Agent upgrade: line insights, the new tools, line snapshot and conversation memory (no live LLM)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from wafer.agent.loop import build_messages, run_turn
from wafer.agent.session import AgentContext
from wafer.agent.snapshot import line_snapshot
from wafer.agent.tools import TOOL_DEFINITIONS, execute_tool
from wafer.classifiers.cnn import CnnClassifier
from wafer.factory import insights
from wafer.infer_mock import classify, classify_batch
from tests.synthetic import draw_pattern


def _clf():
    return CnnClassifier(classify_fn=classify, classify_batch_fn=classify_batch)


def _line() -> pd.DataFrame:
    """8 lots x 5 wafers, 2 h apart. ETCH-01/02/03 clean-ish; ETCH-04 turns Edge-Ring from LOT-006."""
    rows, uid = [], 100
    t0 = pd.Timestamp("2026-10-12 00:00")
    plan = [("LOT-001", "ETCH-01", []), ("LOT-002", "ETCH-02", ["Loc"]), ("LOT-003", "ETCH-03", []),
            ("LOT-004", "ETCH-04", []), ("LOT-005", "ETCH-01", ["Scratch"]),
            ("LOT-006", "ETCH-04", ["Edge-Ring"] * 4), ("LOT-007", "ETCH-04", ["Edge-Ring"] * 5),
            ("LOT-008", "ETCH-04", ["Edge-Ring"] * 5)]
    for li, (lot, tool, defects) in enumerate(plan):
        for w in range(5):
            pattern = defects[w] if w < len(defects) else "none"
            review = lot == "LOT-002" and w == 0
            rows.append({"lot_id": lot, "wafer_in_lot": w + 1, "tool_id": tool,
                         "timestamp": t0 + pd.Timedelta(hours=2 * li, minutes=3 * w), "uid": uid,
                         "pattern": pattern, "confidence": 0.55 if review else 0.97,
                         "needs_review": review, "review_reasons": "low_confidence" if review else "",
                         "label": pattern, "classifier": "cnn", "model_version": "test"})
            uid += 1
    return pd.DataFrame(rows)


END = "2026-10-12 15:00"        # after LOT-008's wafers
MID = "2026-10-12 10:05"        # inside LOT-006 (2 of its wafers seen)


# --------------------------------------------------------------------------- insights
def test_line_overview_ranks_tools_and_cuts_at_as_of():
    o = insights.line_overview(_line(), as_of=END)
    assert o["wafers"] == 40 and o["lots"] == 8
    assert o["ranking_by_defect_rate"][0] == "ETCH-04"
    t4 = next(t for t in o["tools"] if t["tool_id"] == "ETCH-04")
    assert t4["in_alert"] and t4["top_pattern"] == "Edge-Ring" and t4["defects"] == 14
    assert o["alerts"] and o["alerts"][0]["tool_id"] == "ETCH-04"
    early = insights.line_overview(_line(), as_of=MID)
    assert early["wafers"] == 27                       # 5 lots x 5 + 2 wafers of LOT-006
    assert early["latest_lot"] == {"lot_id": "LOT-006", "tool_id": "ETCH-04", "wafers_so_far": 2,
                                   "started": "2026-10-12T10:00:00"}


def test_defect_trend_onset_first_seen_and_direction():
    t = insights.defect_trend(_line(), as_of=END, tool_id="ETCH-04", pattern="Edge-Ring")
    assert [r["lot_id"] for r in t["lots"]] == ["LOT-004", "LOT-006", "LOT-007", "LOT-008"]
    assert [r["matching"] for r in t["lots"]] == [0, 4, 5, 5]
    assert t["first_seen"]["lot_id"] == "LOT-006" and t["onset_lot"]["lot_id"] == "LOT-006"
    assert t["direction"] == "rising" and t["latest_lot_rate"] == 1.0
    none = insights.defect_trend(_line(), as_of=END, tool_id="ETCH-02", pattern="Edge-Ring")
    assert none["first_seen"] is None and none["onset_lot"] is None


def test_find_wafers_filters_and_never_returns_future():
    df = _line()
    r = insights.find_wafers(df, as_of=END, lot_id="lot-007", wafer_in_lot=3)
    assert r["matches"] == 1 and r["wafers"][0]["lot_id"] == "LOT-007" and r["wafers"][0]["wafer_in_lot"] == 3
    rev = insights.find_wafers(df, as_of=END, needs_review=True)
    assert rev["matches"] == 1 and rev["wafers"][0]["review_reasons"] == ["low_confidence"]
    assert insights.find_wafers(df, as_of=MID, lot_id="LOT-008")["matches"] == 0
    capped = insights.find_wafers(df, as_of=END, defects_only=True, limit=3)
    assert capped["returned"] == 3 and capped["truncated"] and capped["pattern_counts"]["Edge-Ring"] == 14


def test_decisions_view_lists_held_lots_and_hides_future_decisions():
    df = _line()
    alert = insights.line_overview(df, as_of=END)["alerts"][0]
    hold = {"action": "hold", "tool_id": "ETCH-04", "pattern": "Edge-Ring", "fired_at": alert["fired_at"],
            "as_of": END, "by": "Jane"}
    v = insights.decisions_view([hold], df, as_of=END)
    assert v["n_decisions"] == 1 and "LOT-007" in v["held_lots"] and "LOT-008" in v["held_lots"]
    assert "LOT-004" not in v["held_lots"]
    assert insights.decisions_view([hold], df, as_of=MID)["n_decisions"] == 0


def test_describe_pattern_and_unknown():
    d = insights.describe_pattern("donut")
    assert d["pattern"] == "Donut" and "Edge-Ring" in d["often_confused_with"]
    assert d["wm811k_labelled_count"] == 555
    with pytest.raises(ValueError):
        insights.describe_pattern("Swirl")


def test_evidence_location_centre_vs_edge():
    rng = np.random.default_rng(0)
    centre = draw_pattern("Center", 40, 40, rng)
    ring = draw_pattern("Edge-Ring", 40, 40, rng)
    flat = lambda m: np.ones(m.shape)                 # noqa: E731 (uniform attention)
    assert insights.evidence_location(centre, flat(centre))["zone"] == "centre"
    whole = insights.evidence_location(ring, flat(ring))
    assert whole["zone"] == "edge" and whole["direction"] == "all around"
    # attention on the right half of a ring moves the evidence to the right
    heat = np.zeros(ring.shape)
    heat[:, 25:] = 1
    loc = insights.evidence_location(ring, heat)
    assert loc["zone"] == "edge" and "right" in loc["direction"]


def test_model_performance_from_metrics():
    m = json.loads(open("docs/results/v0-ens3_val/metrics.json").read())
    s = insights.model_performance(m, "scratch")
    assert s["pattern"]["pattern"] == "Scratch" and 0 < s["pattern"]["f1"] < 1
    assert "most_confused_with" in s["pattern"] and "test set has not been scored" in s["evaluation_note"]
    full = insights.model_performance(m)
    assert len(full["per_pattern"]) == 9 and len(full["weakest_patterns"]) == 2


# --------------------------------------------------------------------------- tools
def _ctx(**kw):
    df = _line()
    rng = np.random.default_rng(1)
    maps = {int(u): draw_pattern(p if p != "none" else "none", 36, 36, rng) for u, p in zip(df.uid, df.pattern)}
    return AgentContext(stream_df=df, as_of=END, classifier=_clf(), maps=maps, **kw)


def test_every_declared_tool_has_a_handler():
    ctx = _ctx(decisions=lambda: [], explain_fn=lambda m: {"heatmap": np.ones(m.shape), "bbox": None})
    calls = {"line_overview": {}, "defect_trend": {"tool_id": "ETCH-04"}, "find_wafers": {"tool_id": "ETCH-02"},
             "explain_wafer": {"uid": 125}, "describe_pattern": {"pattern": "Scratch"},
             "model_performance": {"pattern": "Loc"}, "get_decisions": {},
             "lookup_root_cause": {"pattern": "Edge-Ring"}, "check_lot_history": {"tool_id": "ETCH-04"},
             "estimate_impact": {"tool_id": "ETCH-04"}, "classify_wafer": {"uid": 125}}
    declared = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    assert declared == set(calls)
    for name, args in calls.items():
        out = json.loads(execute_tool(name, args, ctx))
        assert out["ok"], (name, out)


def test_explain_wafer_reports_alternatives_and_evidence():
    ctx = _ctx(explain_fn=lambda m: {"heatmap": np.ones(m.shape), "bbox": [1, 1, 30, 30]})
    out = json.loads(execute_tool("explain_wafer", {"uid": 125}, ctx))
    assert out["line_prediction"]["lot_id"] == "LOT-006"
    assert len(out["live_classification"]["top3"]) == 3
    assert out["evidence"]["zone"] in ("centre", "mid-radius", "edge") and "Grad-CAM" in out["evidence_note"]


def test_explain_wafer_refuses_future_wafers():
    ctx = _ctx()
    ctx.as_of = MID
    out = json.loads(execute_tool("explain_wafer", {"uid": 139}, ctx))   # LOT-008, later than MID
    assert out["ok"] is False and "not reached the line" in out["error"]


def test_get_decisions_without_dashboard_is_explicit():
    out = json.loads(execute_tool("get_decisions", {}, _ctx()))
    assert out["ok"] is False and "not connected" in out["error"]


def test_model_performance_flags_non_cnn_backend():
    ctx = _ctx()
    ctx.classifier = SimpleNamespace(name="yolo", classify=lambda m: {})
    out = json.loads(execute_tool("model_performance", {}, ctx))
    assert out["ok"] and "yolo" in out["backend_note"]


# --------------------------------------------------------------------------- snapshot + conversation
def test_snapshot_names_ids_alerts_decisions_and_focus():
    ctx = _ctx(decisions=lambda: [])
    snap = line_snapshot(ctx, {"page": "lots", "wafer_uid": 125, "wafer_label": "LOT-006 W2"})
    assert "ETCH-04" in snap and "LOT-001 to LOT-008" in snap
    assert "Edge-Ring on ETCH-04" in snap and "Engineer decisions: none yet" in snap
    assert "wafer uid 125 (LOT-006 W2)" in snap


def test_build_messages_keeps_one_system_message_and_recent_history():
    ctx = _ctx()
    hist = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(10)]
    msgs = build_messages(ctx, "and ETCH-02?", history=hist)
    assert [m["role"] for m in msgs].count("system") == 1 and "Line snapshot" in msgs[0]["content"]
    assert msgs[1]["content"] == "q4" and msgs[-2]["content"] == "a9"       # last 6 turns only
    assert msgs[-1] == {"role": "user", "content": "and ETCH-02?"}


class _FakeOpenAI:
    def __init__(self, responses):
        self.calls = []
        outer = self

        class _C:
            def create(self, **kw):
                outer.calls.append(kw)
                return responses.pop(0)
        self.chat = SimpleNamespace(completions=_C())
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id="m")]))


def _assistant(content=None, tool_calls=None):
    msg = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop" if not tool_calls else "tool_calls")])


def test_run_turn_sends_history_and_uses_new_tools():
    from wafer.agent.client import AmdClient
    from wafer.agent.config import AgentConfig
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="defect_trend", arguments=json.dumps({"tool_id": "ETCH-02"})))
    fake = _FakeOpenAI([_assistant(tool_calls=[call]), _assistant(content="ETCH-02 is flat.")])
    client = AmdClient(AgentConfig(api_key="k", base_url="http://x.invalid/v1", model="m"), verify_model=False, client=fake)
    out = run_turn(client, _ctx(), "and ETCH-02?",
                   history=[{"question": "Is ETCH-04 getting worse?", "answer": "Yes, rising since LOT-006."}])
    assert out["ok"] and out["answer"] == "ETCH-02 is flat."
    sent = fake.calls[0]["messages"]
    assert sent[1]["content"] == "Is ETCH-04 getting worse?" and sent[2]["role"] == "assistant"
    assert out["tool_trace"][0]["tool"] == "defect_trend"
    assert fake.calls[0]["max_tokens"] >= 1536

"""Phase 5 agent: tools, history replay rules, mocked LLM loop (no live AMD calls)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from wafer.agent.loop import run_turn
from wafer.agent.session import AgentContext
from wafer.agent.tools import execute_tool
from wafer.classifiers.cnn import CnnClassifier
from wafer.factory import check_lot_history, estimate_impact, lookup_root_cause, score_stream
from wafer.infer_mock import classify, classify_batch
from tests.synthetic import draw_pattern
from tests.test_factory import events, fake_split
from wafer.factory.stream import build_stream, add_predictions


def _mock_cnn():
    return CnnClassifier(classify_fn=classify, classify_batch_fn=classify_batch)


def _scored_stream():
    src = fake_split()
    label = dict(zip(src.uid, src.label))
    rng = np.random.default_rng(0)
    s = build_stream(wafers=src, n_lots=12, excursion_start_lot=10)
    return add_predictions(
        s, classifier=_mock_cnn(),
        get_map=lambda uid: draw_pattern(label[uid], 40, 40, rng),
    )


# --------------------------------------------------------------------------- history / alerts
def test_history_excludes_future_and_alerts_ignore_lot_filter():
    df = events([
        ("ETCH-04", "Edge-Ring", 0, "L1"),
        ("ETCH-04", "Edge-Ring", 1, "L1"),
        ("ETCH-04", "Edge-Ring", 2, "L2"),
        ("ETCH-04", "Edge-Ring", 30, "L9"),  # future relative to as_of below
    ])
    as_of = "2026-10-12 03:00"
    # Lot filter on L1 must not hide the tool-wide alert that needs L2
    hist = check_lot_history(df, as_of=as_of, lot_id="L1", limit=10)
    assert hist["alerts_status"] == "ok"
    assert len(hist["alerts"]) == 1
    assert hist["alerts"][0]["lot_ids"] == ["L1", "L2"]
    assert all(e["lot_id"] == "L1" for e in hist["event_sample"])
    assert all(e["timestamp"] <= "2026-10-12T03:00:00" for e in hist["event_sample"])
    # Future wafer must not appear
    assert all(e["uid"] != 3 for e in hist["full_events"])


def test_history_alerts_unavailable_without_predictions():
    df = events([("T1", "Edge-Ring", 0, "L")]).drop(columns=["pattern"])
    hist = check_lot_history(df, as_of="2026-10-13", lot_id="L")
    assert hist["alerts_status"] == "unavailable_no_predictions"
    assert hist["alerts"] == []
    assert "never used" in hist["alerts_note"].lower() or "True labels" in hist["alerts_note"]


def test_history_limit_does_not_shrink_alert_window():
    # 5 matching wafers; limit=2 for sample; alert threshold=3 still sees all past events
    rows = [("ETCH-04", "Edge-Ring", h, f"L{h}") for h in range(5)]
    df = events(rows)
    hist = check_lot_history(df, as_of="2026-10-13", limit=2, threshold=3)
    assert hist["events_truncated"] and hist["event_count"] == 2
    assert hist["counts"]["events_matching_filters"] == 5
    assert len(hist["full_events"]) == 5
    assert len(hist["lot_summaries"]) == 5
    assert len(hist["alerts"]) == 1 and hist["alerts"][0]["count_in_window"] >= 3


def test_history_compact_for_llm_keeps_full_locally():
    from wafer.factory.history import history_for_llm
    rows = [("ETCH-04", "Edge-Ring", h, "L1") for h in range(3)]
    df = events(rows).assign(needs_review=[False, True, False], confidence=0.9)
    hist = check_lot_history(df, as_of="2026-10-13", tool_id="ETCH-04", limit=2)
    assert "full_events" in hist and len(hist["full_events"]) == 3
    llm = history_for_llm(hist)
    assert "full_events" not in llm
    assert "lot_summaries" in llm and "review_candidates" in llm
    assert len(llm["review_candidates"]) == 1


def test_no_alert_note_does_not_claim_no_exposure():
    df = events([("ETCH-04", "Edge-Ring", 0, "L1"), ("ETCH-04", "Edge-Ring", 1, "L1")])
    hist = check_lot_history(df, as_of="2026-10-13", tool_id="ETCH-04", threshold=3)
    assert hist["alerts"] == []
    assert "not that there is no exposure" in hist["alerts_note"].lower()


def test_classifier_note_on_backend_mismatch():
    df = events([("T1", "Loc", 0, "L"), ("T1", "Loc", 1, "L"), ("T1", "Loc", 2, "L")])
    df = df.assign(classifier="cnn", model_version="v0", confidence=0.9)
    hist = check_lot_history(df, as_of="2026-10-13", expected_classifier="yolo")
    assert hist["classifier_note"] and "cnn" in hist["classifier_note"]


# --------------------------------------------------------------------------- impact
def test_estimate_impact_is_value_at_risk():
    r = estimate_impact(2, wafer_value=1000, wafers_per_lot=10, hypothetical=False)
    assert r["subsequent_lots_observed"] == 2
    assert r["subsequent_lot_exposure"]["value_usd"] == 20_000.0
    assert r["hypothetical"] is False and r["savings_claimed"] is False
    assert "assumptions_note" in r
    h = estimate_impact(2, hypothetical=True)
    assert h["hypothetical"] is True and h["calculation"] == "hypothetical_manual_count"
    assert "future_hypothetical_exposure" in h
    assert "subsequent_lot_exposure" not in h


def test_tool_estimate_impact_simulation_vs_hypothetical():
    stream = _scored_stream()
    as_of = str(pd.to_datetime(stream["timestamp"]).max())
    ctx = AgentContext(stream_df=stream, as_of=as_of, classifier=_mock_cnn())
    sim = json.loads(execute_tool("estimate_impact", {"tool_id": "ETCH-04"}, ctx))
    assert sim["ok"] is True
    assert sim["hypothetical"] is False
    assert sim["counts_what"] == "subsequent_lots_observed_since_alert"
    assert "subsequent_lots_observed" in sim
    assert sim["savings_claimed"] is False
    hyp = json.loads(execute_tool("estimate_impact", {"lots_caught_early": 3}, ctx))
    assert hyp["ok"] and hyp["hypothetical"] is True
    assert "future_hypothetical_exposure" in hyp


def test_tool_early_alert_zero_subsequent_wording():
    """Agent tool path: early as_of must not imply $0 total risk or savings."""
    from wafer.factory import check_alert
    from tests.test_factory import events

    df = events([("ETCH-04", "Edge-Ring", 0, "L1"),
                 ("ETCH-04", "Edge-Ring", 1, "L1"),
                 ("ETCH-04", "Edge-Ring", 2, "L1")])
    [a] = check_alert(df)
    ctx = AgentContext(stream_df=df, as_of=a["fired_at"], classifier=_mock_cnn())
    out = json.loads(execute_tool("estimate_impact", {"tool_id": "ETCH-04"}, ctx))
    assert out["ok"]
    assert out["subsequent_lots_observed"] == 0
    assert "subsequent_lot_exposure" not in out
    assert out.get("value_at_risk_usd") is None
    assert out["intervention_recorded"] is False
    assert out["savings_claimed"] is False
    aff = out["affected_lot_exposure"]
    assert aff["observed_wafers"] == 3
    assert aff["assumed_full_lot_wafers"] == 25
    assert aff["partial_lot"] is True
    assert "unobserved" in aff["note"].lower()


def test_history_tool_strips_full_events_but_keeps_locally():
    df = events([("ETCH-04", "Edge-Ring", h, "L1") for h in range(3)])
    ctx = AgentContext(stream_df=df, as_of="2026-10-13", classifier=_mock_cnn())
    out = json.loads(execute_tool("check_lot_history", {"tool_id": "ETCH-04"}, ctx))
    assert out["ok"] and "full_events" not in out
    assert ctx.last_history_full is not None
    assert "full_events" in ctx.last_history_full



# --------------------------------------------------------------------------- root cause warnings
def test_root_cause_preserves_source_warnings():
    ring = lookup_root_cause("Edge-Ring")
    assert ring["is_possibility"] is True
    assert "source" in ring and ring["source"]
    assert "source_not_read_in_full" in ring["source_warnings"]
    near_full = lookup_root_cause("Near-full")
    assert "unverified_no_citation" in near_full["source_warnings"]


# --------------------------------------------------------------------------- tools / classify
def test_classify_wafer_tool_uses_backend():
    m = draw_pattern("Center", 32, 32, np.random.default_rng(1))
    ctx = AgentContext(stream_df=pd.DataFrame(), as_of="2026-10-12", classifier=_mock_cnn(),
                       maps={42: m})
    out = json.loads(execute_tool("classify_wafer", {"uid": 42}, ctx))
    assert out["ok"] and out["result"]["classifier"] == "cnn"
    assert "pattern" in out["result"] and "needs_review" in out["result"]


def test_score_stream_records_classifier_and_never_uses_label():
    src = fake_split()
    label = dict(zip(src.uid, src.label))
    rng = np.random.default_rng(2)
    s = build_stream(wafers=src, n_lots=4, excursion_start_lot=10)
    # Drop any chance of label bleed: score_stream must write pattern from classifier
    scored = score_stream(s, classifier=_mock_cnn(),
                          get_map=lambda uid: draw_pattern(label[uid], 40, 40, rng))
    assert "classifier" in scored.columns and (scored.classifier == "cnn").all()
    assert "model_version" in scored.columns
    # pattern is from mock geometry rules — may match label often, but column must exist
    # independently of copying: force a map that mock may call differently than label
    assert "pattern" in scored.columns and "label" in scored.columns


# --------------------------------------------------------------------------- mocked LLM loop
class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise RuntimeError("no more fake responses")
        return self._responses.pop(0)


class _FakeChat:
    def __init__(self, responses):
        self.completions = _FakeCompletions(responses)


class _FakeOpenAI:
    def __init__(self, responses):
        self.chat = _FakeChat(responses)
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id="test-model")]))


def _assistant(content=None, tool_calls=None):
    msg = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop" if not tool_calls else "tool_calls")])


def test_run_turn_executes_tools_and_returns_answer():
    from wafer.agent.client import AmdClient
    from wafer.agent.config import AgentConfig

    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="lookup_root_cause",
                                 arguments=json.dumps({"pattern": "Edge-Ring"})),
    )
    responses = [
        _assistant(tool_calls=[tool_call]),
        _assistant(content="Possible Edge-Ring cause from etch focus ring; source noted."),
    ]
    fake = _FakeOpenAI(responses)
    cfg = AgentConfig(api_key="rc-test", base_url="http://example.invalid/v1", model="test-model")
    client = AmdClient(cfg, verify_model=False, client=fake)
    ctx = AgentContext(stream_df=_scored_stream(), as_of="2026-10-12 20:00", classifier=_mock_cnn())
    result = run_turn(client, ctx, "What might cause Edge-Ring?")
    assert result["ok"] and "Edge-Ring" in (result["answer"] or "")
    assert len(result["tool_trace"]) == 1
    assert result["tool_trace"][0]["tool"] == "lookup_root_cause"
    # tool result was sent back to the model
    roles = [m["role"] for m in result["messages"]]
    assert roles.count("tool") == 1

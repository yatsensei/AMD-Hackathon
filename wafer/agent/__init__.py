"""Phase 5: wafer defect pattern assistant (AMD Token Factory + tools).

    from wafer.agent import WaferAgent
    agent = WaferAgent.from_env(stream_df=df, as_of="2026-10-12 18:00")
    first = agent.ask("What is going on with ETCH-04?")
    print(agent.ask("When did it start?",
                    history=[{"question": "What is going on with ETCH-04?", "answer": first["answer"]}])["answer"])

Tools: classify_wafer, check_lot_history, lookup_root_cause, estimate_impact, line_overview,
defect_trend, find_wafers, explain_wafer, describe_pattern, model_performance, get_decisions.
Every question also carries a short line snapshot (wafer/agent/snapshot.py).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from wafer.agent.client import AmdClient
from wafer.agent.config import AgentConfig, load_config
from wafer.agent.loop import run_turn
from wafer.agent.session import AgentContext
from wafer.classifiers import get_classifier
from wafer.classifiers.base import ClassifierBackend


class WaferAgent:
    """Callable interface for CLI and a future Streamlit dashboard."""

    def __init__(self, client: AmdClient, ctx: AgentContext):
        self.client = client
        self.ctx = ctx

    @classmethod
    def from_env(cls, stream_df: pd.DataFrame, as_of: str, *,
                 classifier: ClassifierBackend | str | None = None,
                 verify_model: bool = True,
                 client: Any = None,
                 decisions: Any = None) -> "WaferAgent":
        config = load_config()
        if isinstance(classifier, str) or classifier is None:
            name = classifier if isinstance(classifier, str) else config.classifier
            backend = get_classifier(name)
        else:
            backend = classifier
        amd = AmdClient(config, verify_model=verify_model, client=client)
        ctx = AgentContext(stream_df=stream_df, as_of=as_of, classifier=backend, decisions=decisions)
        return cls(amd, ctx)

    def ask(self, question: str, *, max_rounds: int = 8, history: list[dict] | None = None,
            ui_context: dict | None = None) -> dict:
        """Answer one question. Pass earlier {"question", "answer"} pairs as `history` for follow-ups."""
        return run_turn(self.client, self.ctx, question, max_rounds=max_rounds,
                        history=history, ui_context=ui_context)

    def set_as_of(self, as_of: str) -> None:
        self.ctx.as_of = as_of


def run_query(question: str, *, stream_df: pd.DataFrame, as_of: str,
              classifier: ClassifierBackend | str | None = None,
              verify_model: bool = True, client: Any = None,
              max_rounds: int = 8) -> dict:
    """One-shot helper: build an agent and ask a question."""
    agent = WaferAgent.from_env(stream_df, as_of, classifier=classifier,
                                verify_model=verify_model, client=client)
    return agent.ask(question, max_rounds=max_rounds)


__all__ = ["AgentConfig", "AmdClient", "AgentContext", "WaferAgent", "load_config", "run_query", "run_turn"]

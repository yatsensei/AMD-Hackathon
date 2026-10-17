"""Phase 4: simulated factory stream, alert rule, root-cause lookup and cost impact.

    from wafer.factory import build_stream, check_alert, lookup_root_cause, estimate_impact
    from wafer.factory import check_lot_history, score_stream
"""
from wafer.factory.alerts import check_alert, lots_caught_early
from wafer.factory.history import (check_lot_history, has_predictions, history_for_llm,
                                   stream_as_of, value_at_risk_for_alert)
from wafer.factory.impact import (estimate_impact, impact_from_alert, intervention_recorded,
                                  lot_exposure)
from wafer.factory.root_causes import load_root_causes, lookup_root_cause
from wafer.factory.stream import add_predictions, build_stream, score_stream

__all__ = [
    "add_predictions",
    "build_stream",
    "check_alert",
    "check_lot_history",
    "estimate_impact",
    "has_predictions",
    "history_for_llm",
    "impact_from_alert",
    "intervention_recorded",
    "load_root_causes",
    "lookup_root_cause",
    "lot_exposure",
    "lots_caught_early",
    "score_stream",
    "stream_as_of",
    "value_at_risk_for_alert",
]

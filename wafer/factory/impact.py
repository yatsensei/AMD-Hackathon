"""Exposure figures for the agent's `estimate_impact` tool.

The simulation-derived count is *subsequent lots observed since the alert* on the
same tool (lot start > fired_at), visible at the replay `as_of`. That is not total
fab exposure and not money saved. Affected-lot and future-hypothetical exposures are
separate blocks, included only when calculated. Savings language is allowed only when
the simulation records an intervention.

Partial lots: always distinguish observed wafer counts in the stream from assumed
full-lot value (lots × wafers_per_lot). Never treat unobserved wafers as processed.
"""
from __future__ import annotations

import pandas as pd

from wafer.factory.alerts import lots_caught_early

WAFER_VALUE = 20_000      # USD per wafer; adjustable, an assumption for the demo
WAFERS_PER_LOT = 25       # WM-811K lots hold up to 25 wafers

ASSUMPTIONS_NOTE = (
    "Dollar figures use demo assumptions (default $20k/wafer × wafers_per_lot). "
    "They are exposure estimates under those assumptions, not proven savings. "
    "assumed_full_lot_* multiplies lot count × wafers_per_lot; observed_* counts "
    "only wafers present in the stream as of as_of."
)

COUNTS_WHAT = "subsequent_lots_observed_since_alert"

COUNTS_NOTE = (
    "Counts subsequent lots on the alert's tool whose lot start is after fired_at "
    "and that are visible at as_of. Zero means none have been observed yet — "
    "not that total value at risk is zero, and not that losses were prevented."
)

NO_INTERVENTION_NOTE = (
    "No tool hold or other intervention is recorded in the simulation. "
    "Do not claim the alert stopped production or saved money."
)

PARTIAL_LOT_NOTE = (
    "observed_wafers / observed_value_usd count wafers present in the stream as of as_of. "
    "assumed_full_lot_wafers / assumed_full_lot_value_usd assume every listed lot is full "
    "(lots × wafers_per_lot). Do not describe unobserved wafers as already processed."
)


def _dollarize(n_lots: int, wafer_value: float = WAFER_VALUE,
               wafers_per_lot: int = WAFERS_PER_LOT) -> dict:
    if n_lots < 0 or wafer_value < 0 or wafers_per_lot <= 0:
        raise ValueError("lots and wafer_value must be >= 0, wafers_per_lot > 0")
    wafers = int(n_lots) * int(wafers_per_lot)
    return {
        "lots": int(n_lots),
        "wafers": wafers,
        "value_usd": float(wafers * wafer_value),
        "wafer_value_usd": float(wafer_value),
        "wafers_per_lot": int(wafers_per_lot),
    }


def subsequent_lot_ids(stream_df: pd.DataFrame, alert: dict) -> list[str]:
    """Lot IDs on the alert tool whose first wafer starts after fired_at."""
    df = stream_df[stream_df["tool_id"] == alert["tool_id"]]
    if df.empty:
        return []
    starts = pd.to_datetime(df.groupby("lot_id")["timestamp"].min())
    later = starts[starts > pd.Timestamp(alert["fired_at"])]
    return [str(x) for x in later.sort_values().index.tolist()]


def lot_exposure(stream_df: pd.DataFrame, lot_ids: list[str], *,
                 wafer_value: float = WAFER_VALUE,
                 wafers_per_lot: int = WAFERS_PER_LOT) -> dict:
    """Observed vs assumed-full-lot exposure for a set of lot IDs."""
    ids = [str(x) for x in lot_ids]
    n_lots = len(ids)
    if n_lots == 0:
        return {
            "lot_ids": [],
            "lots": 0,
            "observed_wafers": 0,
            "observed_value_usd": 0.0,
            "assumed_full_lot_wafers": 0,
            "assumed_full_lot_value_usd": 0.0,
            "wafer_value_usd": float(wafer_value),
            "wafers_per_lot": int(wafers_per_lot),
            "partial_lot": False,
            "note": PARTIAL_LOT_NOTE,
        }
    sub = stream_df[stream_df["lot_id"].astype(str).isin(ids)]
    observed = int(len(sub))
    assumed_wafers = n_lots * int(wafers_per_lot)
    return {
        "lot_ids": ids,
        "lots": n_lots,
        "observed_wafers": observed,
        "observed_value_usd": float(observed * wafer_value),
        "assumed_full_lot_wafers": assumed_wafers,
        "assumed_full_lot_value_usd": float(assumed_wafers * wafer_value),
        "wafer_value_usd": float(wafer_value),
        "wafers_per_lot": int(wafers_per_lot),
        "partial_lot": observed < assumed_wafers,
        "note": PARTIAL_LOT_NOTE,
        # Back-compat aliases prefer observed when reporting "seen" exposure.
        "wafers": observed,
        "value_usd": float(observed * wafer_value),
    }


def intervention_recorded(stream_df=None, alert: dict | None = None) -> bool:
    """True only if the simulation explicitly records a hold/intervention."""
    if alert and alert.get("intervention_recorded"):
        return True
    if alert and alert.get("intervention"):
        return True
    if stream_df is not None and hasattr(stream_df, "columns") and "intervention" in stream_df.columns:
        return bool(stream_df["intervention"].astype(bool).any())
    return False


def estimate_impact(lots_caught_early: int, wafer_value: float = WAFER_VALUE,
                    wafers_per_lot: int = WAFERS_PER_LOT, *,
                    hypothetical: bool = False, as_of: str | None = None) -> dict:
    """Dollarize a lot count (assumed full lots). Prefer `impact_from_alert` for simulation."""
    if lots_caught_early < 0 or wafer_value < 0 or wafers_per_lot <= 0:
        raise ValueError("lots_caught_early and wafer_value must be >= 0, wafers_per_lot > 0")

    money = _dollarize(lots_caught_early, wafer_value, wafers_per_lot)
    out = {
        "as_of": as_of,
        "hypothetical": bool(hypothetical),
        "assumptions_note": ASSUMPTIONS_NOTE,
        "intervention_recorded": False,
        "intervention_note": NO_INTERVENTION_NOTE,
        "savings_claimed": False,
        "lots_caught_early": int(lots_caught_early),
        "wafers_saved": money["wafers"] if lots_caught_early > 0 else 0,
        "dollars_saved_usd": money["value_usd"] if lots_caught_early > 0 else 0.0,
        "lots_at_risk": int(lots_caught_early),
        "wafers_at_risk": money["wafers"],
        "value_at_risk_usd": money["value_usd"] if lots_caught_early > 0 else None,
        "wafer_value_usd": float(wafer_value),
        "wafers_per_lot": int(wafers_per_lot),
    }

    if hypothetical:
        out["calculation"] = "hypothetical_manual_count"
        out["future_hypothetical_exposure"] = {
            **money,
            "observed_wafers": None,
            "assumed_full_lot_wafers": money["wafers"],
            "assumed_full_lot_value_usd": money["value_usd"],
            "note": ("Explicitly hypothetical lot count — assumed full lots, not observed "
                     "in the simulation as of as_of."),
        }
    else:
        out["calculation"] = "simulated_subsequent_lots_observed"
        out["counts_what"] = COUNTS_WHAT
        out["subsequent_lots_observed"] = int(lots_caught_early)
        out["counts_note"] = COUNTS_NOTE
        if lots_caught_early > 0:
            out["subsequent_lot_exposure"] = {
                **money,
                "note": ("Assumed full-lot exposure for subsequent lots (no stream rows "
                         "supplied to this helper). Prefer impact_from_alert for observed counts."),
            }
        else:
            out["value_at_risk_usd"] = None
            out["dollars_saved_usd"] = None
            out["wafers_saved"] = None
    return out


def impact_from_alert(stream_df, alert: dict, *, wafer_value: float | None = None,
                      wafers_per_lot: int | None = None, as_of: str | None = None,
                      include_affected_lot_exposure: bool = True) -> dict:
    """Impact from simulation evidence at `as_of` with observed vs assumed-full-lot fields."""
    kwargs = {}
    if wafer_value is not None:
        kwargs["wafer_value"] = wafer_value
    if wafers_per_lot is not None:
        kwargs["wafers_per_lot"] = wafers_per_lot
    wv = kwargs.get("wafer_value", WAFER_VALUE)
    wpl = kwargs.get("wafers_per_lot", WAFERS_PER_LOT)

    later_ids = subsequent_lot_ids(stream_df, alert)
    n = len(later_ids)

    out = estimate_impact(n, hypothetical=False, as_of=as_of, **kwargs)
    out["alert"] = {
        "tool_id": alert.get("tool_id"),
        "pattern": alert.get("pattern"),
        "fired_at": alert.get("fired_at"),
        "fired_lot_id": alert.get("fired_lot_id"),
        "lot_ids": list(alert.get("lot_ids") or []),
    }

    held = intervention_recorded(stream_df, alert)
    out["intervention_recorded"] = held
    out["intervention_note"] = (
        "Simulation records an intervention on this alert."
        if held else NO_INTERVENTION_NOTE
    )
    out["savings_claimed"] = False

    if n > 0:
        exp = lot_exposure(stream_df, later_ids, wafer_value=wv, wafers_per_lot=wpl)
        out["subsequent_lot_exposure"] = {
            **exp,
            "note": PARTIAL_LOT_NOTE + " Subsequent lots whose start is after fired_at.",
        }
        # Prefer observed for headline value_at_risk when partial
        out["value_at_risk_usd"] = exp["observed_value_usd"]
        out["wafers_at_risk"] = exp["observed_wafers"]
        out["dollars_saved_usd"] = None
        out["wafers_saved"] = None
    else:
        out.pop("subsequent_lot_exposure", None)

    if include_affected_lot_exposure:
        lot_ids = list(alert.get("lot_ids") or [])
        if lot_ids:
            exp = lot_exposure(stream_df, lot_ids, wafer_value=wv, wafers_per_lot=wpl)
            out["affected_lot_exposure"] = {
                **exp,
                "note": (PARTIAL_LOT_NOTE + " Lots implicated in the alert window. "
                         "Separate from subsequent lots and from future hypothetical exposure."),
            }

    return out

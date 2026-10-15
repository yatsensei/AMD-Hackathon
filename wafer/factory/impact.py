"""Dollar value of stopping a drifting tool early (the agent's `estimate_impact` tool)."""
from __future__ import annotations

WAFER_VALUE = 20_000      # USD per wafer; adjustable, an assumption for the demo
WAFERS_PER_LOT = 25       # WM-811K lots hold up to 25 wafers


def estimate_impact(lots_caught_early: int, wafer_value: float = WAFER_VALUE,
                    wafers_per_lot: int = WAFERS_PER_LOT) -> dict:
    """Value of the wafers not run on a bad tool because the alert stopped it.

    Plain Python types, so the result is JSON-serialisable for the agent.
    """
    if lots_caught_early < 0 or wafer_value < 0 or wafers_per_lot <= 0:
        raise ValueError("lots_caught_early and wafer_value must be >= 0, wafers_per_lot > 0")
    wafers = int(lots_caught_early) * int(wafers_per_lot)
    return {
        "lots_caught_early": int(lots_caught_early),
        "wafers_saved": wafers,
        "wafer_value_usd": float(wafer_value),
        "wafers_per_lot": int(wafers_per_lot),
        "dollars_saved_usd": float(wafers * wafer_value),
    }

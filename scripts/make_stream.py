#!/usr/bin/env python
"""Phase 4: build the demo factory stream, classify it, and report alerts and impact.

    python scripts/make_stream.py                          # val stream -> streams/stream_val.csv
    python scripts/make_stream.py --split test             # only after the final test evaluation
    python scripts/make_stream.py --mock                   # wafer.infer_mock (no torch; numbers meaningless)
    python scripts/make_stream.py --no-predict             # lots/tools/times only; needs just the split file

Predictions need data/interim/labeled.pkl for the maps (`python scripts/prepare_data.py`).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wafer.env import REPO_ROOT  # noqa: E402
from wafer.factory import add_predictions, build_stream, check_alert, estimate_impact, lots_caught_early  # noqa: E402
from wafer.factory.impact import WAFER_VALUE  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--n-lots", type=int, default=20)
    p.add_argument("--lot-size", type=int, default=25)
    p.add_argument("--excursion-start-lot", type=int, default=10)
    p.add_argument("--excursion-rate", type=float, default=1.0)
    p.add_argument("--baseline-defect-rate", type=float, default=0.02,
                   help="share of defects in healthy lots; negative = the split's natural mix")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window-hours", type=float, default=24)
    p.add_argument("--threshold", type=int, default=3)
    p.add_argument("--wafer-value", type=float, default=WAFER_VALUE)
    p.add_argument("--mock", action="store_true", help="use wafer.infer_mock instead of the CNN")
    p.add_argument("--no-predict", action="store_true", help="write the unscored stream and stop")
    p.add_argument("--out", default=None, help="default: streams/stream_<split>.csv")
    a = p.parse_args(argv)

    stream = build_stream(a.split, n_lots=a.n_lots, lot_size=a.lot_size, excursion_start_lot=a.excursion_start_lot,
                          excursion_rate=a.excursion_rate, seed=a.seed,
                          baseline_defect_rate=None if a.baseline_defect_rate < 0 else a.baseline_defect_rate)
    suffix = "_unscored" if a.no_predict else ""
    out = Path(a.out) if a.out else REPO_ROOT / "streams" / f"stream_{a.split}{suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.no_predict:
        stream.to_csv(out, index=False)
        print(f"wrote {out} ({len(stream)} wafers, {stream.lot_id.nunique()} lots, no predictions)")
        return stream, []
    clf = None
    if a.mock:
        from wafer.infer_mock import classify_batch as clf
    stream = add_predictions(stream, classify_batch=clf)
    stream.to_csv(out, index=False)
    print(f"wrote {out} ({len(stream)} wafers, {stream.lot_id.nunique()} lots, model {stream.model_version.iloc[0]})")

    alerts = check_alert(stream, window_hours=a.window_hours, threshold=a.threshold)
    for al in alerts:
        al["lots_caught_early"] = lots_caught_early(stream, al)
        al["impact"] = estimate_impact(al["lots_caught_early"], wafer_value=a.wafer_value, wafers_per_lot=a.lot_size)
        al["uids"] = f"{len(al['uids'])} wafers"
    print(json.dumps(alerts, indent=2))
    return stream, alerts


if __name__ == "__main__":
    main()

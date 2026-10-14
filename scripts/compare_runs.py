#!/usr/bin/env python
"""Compare training runs on the shared lot-grouped val set (spec 7.3): each run's saved val
outputs + its own calibration, then a paired split-group bootstrap of the macro-F1 and
defect-recall differences against a reference run. Runs must share the val rows (split_protocol lot).

    python scripts/compare_runs.py --ref experiments/e0_cpu_s0_* --runs experiments/e1_nearest_s0_* ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wafer import calibrate, metrics  # noqa: E402


def load(run: Path, meta: pd.DataFrame):
    cal = json.loads((run / "calibration.json").read_text())
    z, uids = np.load(run / "val_logits.npy"), np.load(run / "val_uids.npy")
    m = meta.loc[uids]
    P = calibrate.apply(cal, z)
    return uids, m["label_idx"].to_numpy(), calibrate.decide(cal, P)["pred"], m["split_group"].to_numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", default=None, help="write the table as markdown here")
    a = p.parse_args()
    meta = pd.read_parquet(Path(a.data_dir) / "metadata.parquet").set_index("uid")
    ru, y, ref_pred, groups = load(Path(a.ref), meta)
    rows = []
    for r in map(Path, a.runs):
        u, _, pred, _ = load(r, meta)
        if not np.array_equal(u, ru):
            rows.append({"run": r.name, "note": "different val rows (not lot protocol): not comparable"})
            continue
        f1 = lambda pr: (lambda i: metrics.macro_f1(metrics.confusion(y[i], pr[i])))  # noqa: E731
        rec = lambda pr: (lambda i: metrics.defect_recall_precision(metrics.confusion(y[i], pr[i]))[0])  # noqa: E731
        d = metrics.paired_bootstrap(f1(pred), f1(ref_pred), groups, a.n_boot)
        dr = metrics.paired_bootstrap(rec(pred), rec(ref_pred), groups, a.n_boot)
        rows.append({"run": r.name, "macro_f1": f1(pred)(np.arange(len(y))),
                     "delta_macro_f1": d["point"], "ci": f"[{d['lo']:+.4f}, {d['hi']:+.4f}]",
                     "significant": d["significant"], "delta_defect_recall": dr["point"],
                     "ci_recall": f"[{dr['lo']:+.4f}, {dr['hi']:+.4f}]"})
    df = pd.DataFrame(rows)
    print(f"reference {Path(a.ref).name}: macro-F1 {metrics.macro_f1(metrics.confusion(y, ref_pred)):.4f}")
    print(df.to_string(index=False))
    if a.out:
        try:
            text = df.to_markdown(index=False)
        except ImportError:                                    # tabulate not installed
            text = df.to_string(index=False)
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()

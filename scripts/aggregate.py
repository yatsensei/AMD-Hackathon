#!/usr/bin/env python
"""Rebuild docs/results/leaderboard.csv from every experiments/*/summary.json (spec 5.6), and
print mean ± SD per config. Runs never write the CSV themselves, so parallel runs can't conflict."""
import argparse
import json
from pathlib import Path

import pandas as pd

COLS = ["run_id", "experiment", "config", "model", "encoding", "loss", "sampler", "aug", "seed", "val_macro_f1",
        "val_defect_macro_f1", "val_bal_acc", "val_defect_recall", "val_defect_precision",
        "val_macro_f1_argmax_raw", "val_ece_defect", "val_review_rate", "cpu_ms", "epochs", "train_minutes",
        "gpu", "git_commit", "dirty"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiments", default="experiments")
    p.add_argument("--out", default="docs/results/leaderboard.csv")
    a = p.parse_args()
    rows = [json.loads(f.read_text()) for f in sorted(Path(a.experiments).glob("*/summary.json"))]
    if not rows:
        raise SystemExit(f"no summary.json under {a.experiments}/")
    df = pd.DataFrame(rows).reindex(columns=COLS)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["experiment", "config", "seed"]).to_csv(a.out, index=False, float_format="%.5f")
    g = df.groupby("config").agg(n=("seed", "size"), macro_f1=("val_macro_f1", "mean"), sd=("val_macro_f1", "std"),
                                 defect_f1=("val_defect_macro_f1", "mean"), recall=("val_defect_recall", "mean"),
                                 cpu_ms=("cpu_ms", "mean"), minutes=("train_minutes", "mean"))
    print(g.sort_values("macro_f1", ascending=False).round(4).to_string())
    print(f"\nwrote {a.out} ({len(df)} runs)")


if __name__ == "__main__":
    main()

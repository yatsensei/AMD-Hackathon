#!/usr/bin/env python
"""E0b classical baseline (spec 6.1): hand-crafted features + HistGradientBoosting, CPU only.

Writes a predictions CSV (uid, prob_<class>) for val and test rows plus a prior sidecar, so it is
scored by exactly the same code as the CNN:
    python scripts/e0b_classical.py
    python -m wafer.evaluate --predictions experiments/e0b/predictions.csv     # val
Test rows are predicted but never scored here (only the single final run may score test).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wafer.constants import CLASSES  # noqa: E402
from wafer.features import all_features  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--out", default="experiments/e0b")
    p.add_argument("--max-iter", type=int, default=300)
    a = p.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier

    t0 = time.time()
    z = np.load(Path(a.data_dir) / "wafers_s64.npz")
    meta = pd.read_parquet(Path(a.data_dir) / "metadata.parquet")
    assert np.array_equal(z["uid"], meta["uid"].to_numpy())
    X = all_features(z["X_area"])
    y = z["y"]
    print(f"features {X.shape} in {time.time() - t0:.0f}s")
    split = meta["split"].to_numpy()
    tr = split == "train"
    clf = HistGradientBoostingClassifier(max_iter=a.max_iter, learning_rate=0.1, class_weight="balanced",
                                         early_stopping=False, random_state=0)
    clf.fit(X[tr], y[tr])
    print(f"fit in {time.time() - t0:.0f}s")
    keep = split != "train"
    P = np.zeros((keep.sum(), len(CLASSES)))
    P[:, clf.classes_] = clf.predict_proba(X[keep])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(P, columns=[f"prob_{c}" for c in CLASSES])
    df.insert(0, "uid", z["uid"][keep])
    df.to_csv(out / "predictions.csv", index=False, float_format="%.6g")
    counts = np.bincount(y[tr], minlength=len(CLASSES)).astype(float)
    (out / "predictions.prior.json").write_text(json.dumps({
        "pi_natural": (counts / counts.sum()).tolist(),
        "pi_train_eff": [1 / len(CLASSES)] * len(CLASSES),     # class_weight="balanced"
        "model": f"HistGradientBoostingClassifier(max_iter={a.max_iter}, class_weight=balanced) on wafer.features"}))
    print(f"wrote {out / 'predictions.csv'} ({len(df)} rows) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

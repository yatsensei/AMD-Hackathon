"""A small synthetic stand-in for LSWMD.pkl with the same columns, nested-array quirks and a few
deliberate defects (invalid values, an empty map, a missing lot, exact duplicates)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wafer.constants import CLASSES

SHAPES = [(26, 26), (30, 34), (45, 48), (64, 60)]


def circle_wafer(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """(map with 1 inside the wafer, normalised radius r)."""
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot((yy - (h - 1) / 2) / (h / 2), (xx - (w - 1) / 2) / (w / 2))
    return (r <= 1.0).astype(np.uint8), r


def draw_pattern(label: str, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    m, r = circle_wafer(h, w)
    inside = m == 1
    yy, xx = np.mgrid[:h, :w]
    theta = np.arctan2(yy - h / 2, xx - w / 2)
    fail = rng.random((h, w)) < 0.01                                 # background noise
    if label == "Center":
        fail |= r < 0.3
    elif label == "Donut":
        fail |= (r > 0.35) & (r < 0.6)
    elif label == "Edge-Ring":
        fail |= r > 0.85
    elif label == "Edge-Loc":
        a = rng.uniform(-np.pi, np.pi)
        fail |= (r > 0.75) & (np.abs(np.angle(np.exp(1j * (theta - a)))) < 0.5)
    elif label == "Loc":
        cy, cx = rng.uniform(0.3, 0.7) * h, rng.uniform(0.3, 0.7) * w
        fail |= np.hypot(yy - cy, xx - cx) < min(h, w) * 0.1
    elif label == "Near-full":
        fail |= rng.random((h, w)) < 0.9
    elif label == "Random":
        fail |= rng.random((h, w)) < 0.3
    elif label == "Scratch":
        n = min(h, w)
        i = np.arange(n // 5, 4 * n // 5)
        fail[i, i + rng.integers(-3, 4)] = True
    m[inside & fail] = 2
    return m


def nested(x):
    return np.array([[x]], dtype=object)


def make_raw_df(n_lots: int = 240, wafers_per_lot: int = 12, unlabeled_frac: float = 0.3,
                seed: int = 0) -> pd.DataFrame:
    """Lots carry a dominant class (like the real data, defects cluster by lot)."""
    rng = np.random.default_rng(seed)
    probs = np.array([0.08, 0.06, 0.08, 0.08, 0.08, 0.05, 0.06, 0.06, 0.45])
    rows = []
    for lot in range(n_lots):
        dominant = CLASSES[lot % len(CLASSES)] if lot < 4 * len(CLASSES) else rng.choice(CLASSES, p=probs)
        h, w = SHAPES[rng.integers(len(SHAPES))]
        for i in range(wafers_per_lot):
            label = dominant if rng.random() < 0.6 else "none"
            m = draw_pattern(label, h, w, rng)
            labelled = rng.random() >= unlabeled_frac
            rows.append({
                "waferMap": m,
                "dieSize": float((m > 0).sum()),
                "lotName": f"lot{lot + 1}",
                "waferIndex": float(i + 1),
                "trianTestLabel": nested("Training" if rng.random() < 0.3 else "Test") if labelled
                else np.zeros((0, 0)),
                "failureType": nested(label) if labelled else np.array([], dtype=object),
            })
    df = pd.DataFrame(rows)

    # Deliberate defects the pipeline must handle.
    extra = [
        {"waferMap": np.full((20, 20), 3, np.uint8), "dieSize": 1.0, "lotName": "lotbad",
         "waferIndex": 1.0, "trianTestLabel": nested("Test"), "failureType": nested("none")},
        {"waferMap": np.zeros((20, 20), np.uint8), "dieSize": 0.0, "lotName": "lotempty",
         "waferIndex": 1.0, "trianTestLabel": nested("Test"), "failureType": nested("Loc")},
        {"waferMap": draw_pattern("Center", 26, 26, rng), "dieSize": 1.0, "lotName": None,
         "waferIndex": 3.0, "trianTestLabel": nested("Test"), "failureType": nested("Center")},
    ]
    # Exact duplicates of labelled maps, placed in new single-wafer lots (the pipeline must link
    # them to the original's split group); one copy carries a conflicting label.
    labelled_idx = [i for i, ft in enumerate(df["failureType"]) if np.asarray(ft).size]
    for k, src in enumerate(rng.choice(labelled_idx, size=min(30, len(labelled_idx) // 2), replace=False)):
        lab = str(np.asarray(df.at[src, "failureType"]).ravel()[0])
        if k == 0:
            lab = "Random" if lab != "Random" else "Loc"
        extra.append({"waferMap": df.at[src, "waferMap"].copy(), "dieSize": df.at[src, "dieSize"],
                      "lotName": f"duplot{k}", "waferIndex": 1.0, "trianTestLabel": nested("Test"),
                      "failureType": nested(lab)})
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True)

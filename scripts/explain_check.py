#!/usr/bin/env python
"""Explainability check (spec 8 / 7.5) on validation wafers only.

1. docs/results/<model>_val/gradcam_examples.png: one confidently correct val wafer per class,
   with the Grad-CAM heatmap and the defect box that `explain()` returns.
2. Synthetic-insertion benchmark (spec 8.4): paste a known pattern with a known box onto clean
   val `none` wafers (a 1-die scratch line, a Loc-like blob, an Edge-Loc-like arc), then report,
   on insertions classified as the intended class: pointing-game hit rate (heatmap peak inside the
   true box) and median box IoU. Sanity numbers about *where* the model looks, not headline metrics.

    python scripts/explain_check.py --model models/v0
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wafer.constants import CLASSES  # noqa: E402
from wafer.data.access import get_wafer_map, load_split  # noqa: E402
from wafer.infer import load_model  # noqa: E402
from wafer.viz import render_heatmap  # noqa: E402

KINDS = {"Scratch": "scratch", "Loc": "blob", "Edge-Loc": "edge_arc"}


def _geometry(m):
    ys, xs = np.nonzero(m > 0)
    cy, cx = ys.mean(), xs.mean()
    R = min(np.ptp(ys), np.ptp(xs)) / 2
    yy, xx = np.mgrid[:m.shape[0], :m.shape[1]]
    return cy, cx, R, np.hypot(yy - cy, xx - cx) / max(R, 1), np.arctan2(yy - cy, xx - cx)


def insert(m, kind, rng):
    """Return (map with the pattern, true box [x0,y0,x1,y1]) or None if it didn't fit."""
    m = m.copy()
    cy, cx, R, r, th = _geometry(m)
    mask = np.zeros(m.shape, np.uint8)
    if kind == "scratch":
        a = rng.uniform(0, np.pi)
        ox, oy = cx + rng.uniform(-0.3, 0.3) * R, cy + rng.uniform(-0.3, 0.3) * R
        half = rng.uniform(0.3, 0.5) * R
        p0 = (int(round(ox - half * np.cos(a))), int(round(oy - half * np.sin(a))))
        p1 = (int(round(ox + half * np.cos(a))), int(round(oy + half * np.sin(a))))
        cv2.line(mask, p0, p1, 1, thickness=1)
    elif kind == "blob":
        rr, a = rng.uniform(0.2, 0.55), rng.uniform(-np.pi, np.pi)
        by, bx = cy + rr * R * np.sin(a), cx + rr * R * np.cos(a)
        yy, xx = np.mgrid[:m.shape[0], :m.shape[1]]
        mask = (np.hypot(yy - by, xx - bx) <= rng.uniform(0.1, 0.16) * R).astype(np.uint8)
    else:  # edge arc
        a = rng.uniform(-np.pi, np.pi)
        d = np.abs(np.angle(np.exp(1j * (th - a))))
        mask = ((r > 0.82) & (d < np.radians(rng.uniform(20, 35)))).astype(np.uint8)
    mask &= (m > 0).astype(np.uint8)
    if mask.sum() < 4:
        return None
    m[mask == 1] = 2
    ys, xs = np.nonzero(mask)
    return m, [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]) + 1)
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]) + 1)
    inter = ix * iy
    area = lambda r: (r[2] - r[0] + 1) * (r[3] - r[1] + 1)  # noqa: E731
    return inter / (area(a) + area(b) - inter)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/v0")
    p.add_argument("--n", type=int, default=300, help="clean val wafers per pattern")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--cam-block", type=int, default=None, help="small_cnn only: override the Grad-CAM block (1-4)")
    p.add_argument("--out-suffix", default="")
    a = p.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    clf = load_model(a.model, "cpu")
    if a.cam_block:
        for net in clf.members:
            net._cam_block = a.cam_block
    out = Path("docs/results") / f"{Path(a.model).name}_val{a.out_suffix}"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    split = load_split()
    val = split[split["split"] == "val"]

    # 1. one confident correct example per class
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(10, 10.5))
    for ax, c in zip(axes.ravel(), CLASSES):
        ax.axis("off")
        cand = val[(val["label"] == c) & (val["orig_h"] >= 25)].sample(frac=1, random_state=0)
        for uid in cand["uid"][:40]:
            m = get_wafer_map(uid)
            r = clf.classify(m)
            if r["pattern"] == c and r["confidence"] > 0.8:
                e = clf.explain(m)
                ax.imshow(render_heatmap(m, e["heatmap"], e["bbox"]), interpolation="nearest")
                ax.set_title(f"{c} ({r['confidence']:.2f})", fontsize=10)
                break
    fig.suptitle(f"{Path(a.model).name}: Grad-CAM heatmap + defect box on correctly classified val wafers")
    fig.tight_layout()
    fig.savefig(out / "gradcam_examples.png", dpi=120)
    plt.close(fig)

    # 2. synthetic insertion benchmark
    clean = val[(val["label"] == "none") & (val["orig_h"] >= 30) & (val["orig_w"] >= 30)
                & (val["n_fail"] <= 0.03 * val["n_die"])].sample(n=a.n, random_state=0)["uid"].to_numpy()
    res, gallery = {}, []
    for target, kind in KINDS.items():
        hit = point = 0
        ious, n = [], 0
        for uid in clean:
            ins = insert(get_wafer_map(uid), kind, rng)
            if ins is None:
                continue
            m, gt = ins
            n += 1
            e = clf.explain(m)
            if e["pattern"] != target:
                continue
            hit += 1
            py, px = np.unravel_index(np.argmax(e["heatmap"]), e["heatmap"].shape)
            point += int(gt[0] - 1 <= px <= gt[2] + 1 and gt[1] - 1 <= py <= gt[3] + 1)
            ious.append(iou(e["bbox"], gt))
            if len([g for g in gallery if g[0] == target]) < 3:
                gallery.append((target, m, e, gt))
        res[target] = {"inserted": n, "classified_as_intended": hit, "classification_rate": hit / max(n, 1),
                       "pointing_game": point / max(hit, 1), "median_box_iou": float(np.median(ious)) if ious else 0.0,
                       "box_iou_ge_0.3": float(np.mean(np.array(ious) >= 0.3)) if ious else 0.0}
        print(target, res[target])
    (out / "localisation.json").write_text(json.dumps(
        {"model": a.model, "note": "synthetic patterns pasted on clean val wafers; sanity numbers, not headline",
         "results": res}, indent=2))

    fig, axes = plt.subplots(3, 3, figsize=(10, 10.5))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (t, m, e, gt) in zip(axes.ravel(), gallery):
        img = render_heatmap(m, e["heatmap"], e["bbox"])
        cv2.rectangle(img, (gt[0], gt[1]), (gt[2], gt[3]), (255, 255, 0), 1)
        ax.imshow(img, interpolation="nearest")
        ax.set_title(f"inserted {t}: box IoU {iou(e['bbox'], gt):.2f}", fontsize=9)
    fig.suptitle("Synthetic insertions: white = model's box, yellow = true box")
    fig.tight_layout()
    fig.savefig(out / "localisation_examples.png", dpi=120)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()

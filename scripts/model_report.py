#!/usr/bin/env python
"""Model assessment charts (validation only) -> docs/results/report/*.png + REPORT.md.

    python scripts/model_report.py --model models/v0-ens3

Recomputes the model's validation predictions (a few minutes on CPU) and reads the saved
experiment results (docs/results/*_val/metrics.json, experiments/*/{summary.json,history.csv}).
Never touches the test split.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wafer import calibrate, metrics  # noqa: E402
from wafer.constants import CLASSES, NONE_IDX  # noqa: E402

# Reference palette (dataviz skill, light mode)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"           # categorical slots 1-3 (blue, orange, aqua)
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 12,
        "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "legend.frameon": False, "lines.linewidth": 2,
    })
    return plt


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)


def predictions(model_dir: Path, data_dir: Path):
    """Scored val predictions plus raw (pre-calibration) probabilities, cached under data/."""
    from wafer.evaluate import Scored
    cache = Path("data") / f"report_cache_{model_dir.name}.npz"
    s = Scored(model_dir, data_dir, "val", "cpu", cache=cache)
    return s, calibrate.softmax(s.z)        # calibrated is s.P; raw = plain softmax of the outputs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/v0-ens3")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--out", default="docs/results/report")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plt = style()
    name = Path(a.model).name
    s, P_raw = predictions(Path(a.model), Path(a.data_dir))
    y, d = s.y, s.d
    pred, conf = d["pred"], d["confidence"]
    res = json.loads((Path("docs/results") / f"{name}_val" / "metrics.json").read_text())
    pc = pd.DataFrame(res["per_class"]).set_index("class")
    ci = res["ci"]["per_class_f1"]

    # 1. Per-class precision / recall / F1 (F1 with 95% CI by lot)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(CLASSES))
    w = 0.26
    for k, (col, c) in enumerate([("precision", S1), ("recall", S2), ("f1", S3)]):
        ax.bar(x + (k - 1) * w, pc.loc[CLASSES, col], w - 0.03, color=c, label=col.capitalize() if col != "f1" else "F1")
    lo = [ci[c]["point"] - ci[c]["lo"] for c in CLASSES]
    hi = [ci[c]["hi"] - ci[c]["point"] for c in CLASSES]
    ax.errorbar(x + w, pc.loc[CLASSES, "f1"], yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    ax.set_xticks(x, [f"{c}\n(n={int(pc.loc[c, 'support'])})" for c in CLASSES], fontsize=8)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("score")
    ax.set_title(f"Per-class precision, recall and F1 ({name}, validation; F1 whiskers = 95% CI by lot)")
    ax.legend(ncol=3, loc="lower left")
    save(fig, out, "per_class.png")

    # 2. Confusion matrix (row-normalised = recall view)
    from matplotlib.colors import LinearSegmentedColormap
    cm = np.asarray(res["confusion"])
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    ax.grid(False)
    ax.imshow(norm, cmap=LinearSegmentedColormap.from_list("seq", SEQ), vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i, j]:.2f}\n{cm[i, j]}", ha="center", va="center", fontsize=6.5,
                        color="#ffffff" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=40, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (share of each true class; count below)")
    save(fig, out, "confusion.png")

    # 3. Calibration before vs after, on wafers predicted defective (where confidence matters;
    #    the 85% easy `none` calls near 100% confidence would otherwise dominate the picture)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="perfect calibration")
    raw_pred = P_raw.argmax(1)
    for cf, ok, label, c in [
        (P_raw.max(1)[raw_pred != NONE_IDX], (raw_pred == y)[raw_pred != NONE_IDX], "before (raw softmax)", S2),
        (conf[pred != NONE_IDX], (pred == y)[pred != NONE_IDX], "after (prior correction + temperature)", S1),
    ]:
        order = np.argsort(cf)
        bins = np.array_split(order, 12)
        ax.plot([cf[b].mean() for b in bins], [ok[b].mean() for b in bins], marker="o", ms=5, color=c,
                label=f"{label}: ECE {metrics.ece(cf, ok):.3f}, {len(cf):,} wafers")
    ax.set_xlim(0.3, 1.01)
    ax.set_ylim(0.3, 1.01)
    ax.set_xlabel("model confidence")
    ax.set_ylabel("actual accuracy")
    ax.set_title("Calibration on defect calls: 90% confident ≈ 90% right")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, out, "calibration.png")

    # 4. Coverage vs accuracy on predicted defects
    pdm = pred != NONE_IDX
    cr = metrics.coverage_risk(conf[pdm], y[pdm], pred[pdm], coverages=np.linspace(0.5, 1.0, 26))
    tau = s.cal["tau"]
    cov_tau = float((conf[pdm] >= tau).mean())
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(cr["coverage"], cr["accuracy"], color=S1, label="accuracy")
    ax.plot(cr["coverage"], cr["macro_f1"], color=S2, label="macro-F1")
    ax.axvline(cov_tau, color=INK2, lw=1, ls=":")
    ax.text(cov_tau, 0.86, f"  review threshold:\n  {cov_tau:.0%} auto-handled", color=INK2, fontsize=8)
    ax.set_xlabel("share of defect calls the model handles itself (most confident first)")
    ax.set_ylabel("score")
    ax.set_ylim(0.84, 1.005)
    ax.set_title("Confidence is useful: the most confident defect calls are almost always right")
    ax.legend(loc="lower left")
    save(fig, out, "coverage.png")

    # 5. Every experiment on validation
    rows = []
    for f in sorted(Path("experiments").glob("*/summary.json")):
        sm = json.loads(f.read_text())
        rows.append({"config": sm["config"], "macro_f1": sm["val_macro_f1"], "protocol": sm.get("split_protocol", "lot")})
    runs = pd.DataFrame(rows)
    runs["config"] = runs["config"].replace({"e0_cpu_fp32": "e0_cpu"})
    labels = {"e0_cpu": "Small CNN, 30 ep (5 seeds)", "e0_60ep": "Small CNN, 60 epochs", "e3_resnet18s": "ResNet-18",
              "e3b_rotation": "+ rotation aug (3 seeds)", "e2_focal": "Focal loss", "e1_nearest": "Nearest encoding",
              "eleak_wafer_random": "Random split (leak check)*"}
    ens = {"3-CNN ensemble (v0-ens3)": "v0-ens3", "5-CNN ensemble": "v0-ens5", "1 CNN + TTA": "v0-tta",
           "Classical baseline (E0b)": "e0b"}
    items = []
    for lab, key in ens.items():
        r = json.loads((Path("docs/results") / f"{key}_val" / "metrics.json").read_text())
        items.append((lab, r["macro_f1"], r["ci"]["macro_f1"]["lo"], r["ci"]["macro_f1"]["hi"], None))
    for cfg, g in runs.groupby("config"):
        if cfg in labels:
            items.append((labels[cfg], g["macro_f1"].mean(), None, None, g["macro_f1"].to_numpy()))
    items.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for i, (lab, m, l_, h_, seeds) in enumerate(items):
        c = S1 if "ensemble" in lab or "TTA" in lab else (S2 if "baseline" in lab else S3)
        if l_ is not None:
            ax.plot([l_, h_], [i, i], color=c, lw=2, solid_capstyle="round")
        if seeds is not None and len(seeds) > 1:
            ax.scatter(seeds, [i] * len(seeds), s=14, color=c, alpha=0.5, zorder=2)
        ax.scatter([m], [i], s=60, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(m, i + 0.28, f"{m:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(items)), [t[0] for t in items])
    ax.set_xlabel("validation macro-F1")
    ax.set_title("Every experiment (lines = 95% CI by lot; small dots = individual seeds)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=S1, label="ensemble / TTA"),
                       Line2D([], [], marker="o", ls="", color=S3, label="single model variants"),
                       Line2D([], [], marker="o", ls="", color=S2, label="classical baseline")], loc="lower right", fontsize=8)
    fig.text(0.01, 0.01, "* scored on its own random val split, so not directly comparable", fontsize=7, color=INK2)
    save(fig, out, "experiments.png")

    # 6. Training curves (raw argmax val macro-F1, logged every 5-10 epochs)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for pat, lab, c in [("e0_cpu*", "Small CNN, 30 epochs (each seed)", S1), ("e0_60ep*", "Small CNN, 60 epochs", S2),
                        ("e3_resnet18s*", "ResNet-18, 30 epochs", S3)]:
        first = True
        for f in sorted(Path("experiments").glob(f"{pat}/history.csv")):
            h = pd.read_csv(f).dropna(subset=["val_macro_f1"])
            ax.plot(h["epoch"], h["val_macro_f1"], color=c, marker="o", ms=4, alpha=0.85 if first else 0.45,
                    label=lab if first else None)
            first = False
    ax.set_xlabel("epoch")
    ax.set_ylabel("val macro-F1 (before calibration)")
    ax.set_title("Training curves: all converge to a similar level")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, out, "training_curves.png")

    # 7. Robustness: wafer size and noise
    area = (s.meta["orig_h"] * s.meta["orig_w"]).to_numpy()
    q = pd.qcut(area, 4)
    size_rows = []
    for k, b in enumerate(q.categories):
        m_ = (area > b.left) & (area <= b.right)
        size_rows.append((f"Q{k + 1}\n{int(max(b.left, 0)) + 1}–{int(b.right)} dies", metrics.macro_f1(metrics.confusion(y[m_], pred[m_]))))
    sl = pd.read_csv("docs/results/v0_val/slices.csv")
    noise = sl[sl["slice"] == "pixel_flip_noise"]
    base_v0 = json.loads(Path("docs/results/v0_val/metrics.json").read_text())["macro_f1"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.bar(range(4), [r[1] for r in size_rows], color=S1, width=0.6)
    for i, r in enumerate(size_rows):
        ax.text(i, r[1] + 0.005, f"{r[1]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(4), [r[0] for r in size_rows], fontsize=8)
    ax.set_ylim(0.7, 0.97)
    ax.set_ylabel("macro-F1")
    ax.set_title(f"By wafer size ({name})")
    ax = axes[1]
    xs = [0] + [float(b.strip("%")) for b in noise["bin"]]
    ys = [base_v0] + list(noise["macro_f1"])
    ax.plot(xs, ys, color=S2, marker="o", ms=6)
    for xx, yy in zip(xs, ys):
        ax.text(xx, yy + 0.003, f"{yy:.3f}", ha="center", fontsize=8)
    ax.set_xlabel("% of dies flipped pass↔fail at random")
    ax.set_ylim(0.85, 0.92)
    ax.set_title("Robustness to noise (v0)")
    save(fig, out, "robustness.png")

    # 8. Review routing per class
    rr = res["review_rate_per_class"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    vals = [rr[c] for c in CLASSES]
    ax.bar(range(len(CLASSES)), vals, color=S1, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.003, f"{v:.1%}", ha="center", fontsize=8)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
    ax.set_ylabel("share sent to an engineer")
    ax.set_title(f"needs_review: which wafers the model hands to a person ({res['review_rate']:.1%} overall)")
    save(fig, out, "review_rate.png")

    # 9. Localisation sanity (synthetic insertions)
    loc = json.loads(Path("docs/results/v0_val/localisation.json").read_text())["results"]
    pats = list(loc)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(pats))
    for k, (key, lab, c) in enumerate([("classification_rate", "classified correctly", S1),
                                       ("pointing_game", "heatmap peak on defect", S2),
                                       ("box_iou_ge_0.3", "box overlaps defect (IoU ≥ 0.3)", S3)]):
        v = [loc[p_][key] for p_ in pats]
        ax.bar(x + (k - 1) * 0.26, v, 0.23, color=c, label=lab)
    ax.set_xticks(x, [f"inserted {p_}" for p_ in pats])
    ax.set_ylim(0, 1.08)
    ax.set_title("Explanations: synthetic defects pasted onto clean val wafers (v0)")
    ax.legend(loc="lower left", fontsize=8)
    save(fig, out, "localisation.png")

    summary = {"model": name, "val_wafers": int(len(y)), "macro_f1": res["macro_f1"], "ci": res["ci"]["macro_f1"],
               "accuracy": res["accuracy"], "balanced_accuracy": res["balanced_accuracy"],
               "defect_recall": res["defect_recall"], "defect_precision": res["defect_precision"],
               "ece_before": metrics.ece(P_raw.max(1), P_raw.argmax(1) == y), "ece_after": res["ece"],
               "review_rate": res["review_rate"], "auto_handled_defect_share": cov_tau,
               "size_quartile_macro_f1": [r[1] for r in size_rows]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()

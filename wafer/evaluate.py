"""Evaluate a model folder (spec Section 7). Validation by default; test only with --final.

    python -m wafer.evaluate --model models/v0                         # -> docs/results/v0_val/
    python -m wafer.evaluate --final --model models/v1 \\
           --comparators models/v0 [other model folders]                # -> docs/results/final/ (ONCE)

--final refuses to run if docs/results/final/metrics.json already exists (spec Rule 0.1).
Comparators are scored on the same wafers with their own val-fitted calibration and compared
with a paired lot(split-group) bootstrap; they are never candidates to choose between.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from wafer.env import REPO_ROOT, configure_rocm_env, env_info

configure_rocm_env()

import torch  # noqa: E402

from wafer import calibrate, metrics  # noqa: E402
from wafer.constants import CLASSES, NONE_IDX  # noqa: E402
from wafer.data.augment import apply_d4  # noqa: E402
from wafer.data.dataset import WaferData  # noqa: E402
from wafer.infer import WaferClassifier, load_members  # noqa: E402
from wafer.predict import forward_base, predict_rows  # noqa: E402

log = logging.getLogger("wafer.evaluate")
FINAL_DIR = REPO_ROOT / "docs" / "results" / "final"


class Scored:
    """A model's calibrated outputs on one split."""

    def __init__(self, model_dir: Path, data_dir: Path, split: str, device: str):
        self.dir = model_dir
        self.cfg = yaml.safe_load((model_dir / "config.yaml").read_text())
        self.cal = json.loads((model_dir / "calibration.json").read_text())
        self.data = WaferData(data_dir / f"wafers_s{self.cfg['size']}.npz", data_dir / "metadata.parquet",
                              encoding=self.cfg["encoding"], device=device, aux=self.cfg.get("aux_features", False))
        self.idx = self.data.indices(split)
        self.members = load_members(model_dir, self.cfg, device)
        self.tta = bool(self.cfg.get("tta", False))
        z = calibrate.combine_members([predict_rows(m, self.data, self.idx, tta=self.tta) for m in self.members],
                                      self.tta)
        self.P = calibrate.apply(self.cal, z)
        self.d = calibrate.decide(self.cal, self.P)
        self.y = self.data.y_np[self.idx]
        self.meta = self.data.meta.iloc[self.idx].reset_index(drop=True)
        self.groups = (self.meta["split_group"] if "split_group" in self.meta else self.meta["lotName"]).to_numpy()


@torch.no_grad()
def _member_preds(s: Scored, transform) -> np.ndarray:
    """Decisions after applying `transform` to each base batch (members combined, calibrated)."""
    outs = []
    for m in s.members:
        m.eval()
        chunks = []
        for i in range(0, len(s.idx), 4096):
            base, _, aux = s.data.batch(s.idx[i:i + 4096])
            chunks.append(forward_base(m, transform(base), s.data.encoding, aux, tta=s.tta).cpu().numpy())
        outs.append(np.concatenate(chunks))
    P = calibrate.apply(s.cal, calibrate.combine_members(outs, s.tta))
    return calibrate.decide(s.cal, P)["pred"]


def slices(s: Scored) -> pd.DataFrame:
    rows = []
    pred = s.d["pred"]

    def add(kind, name, mask):
        if mask.sum():
            cm = metrics.confusion(s.y[mask], pred[mask])
            rows.append({"slice": kind, "bin": str(name), "n": int(mask.sum()), "macro_f1": metrics.macro_f1(cm),
                         "defect_recall": metrics.defect_recall_precision(cm)[0]})

    area = (s.meta["orig_h"] * s.meta["orig_w"]).to_numpy()
    for b in pd.qcut(area, 4, duplicates="drop").categories:
        add("orig_size_quartile", b, ((area > b.left) & (area <= b.right)))
    die = s.meta["dieSize"].to_numpy()
    for b in pd.qcut(die, 4, duplicates="drop").categories:
        add("dieSize_quartile", b, ((die > b.left) & (die <= b.right)))
    wi = s.meta["waferIndex"].to_numpy()
    add("waferIndex", "1-12", (wi >= 1) & (wi <= 12))
    add("waferIndex", "13-25", wi >= 13)

    preds = [_member_preds(s, lambda b, t=t: apply_d4(b, t)) for t in range(8)]
    consistent = np.all([p == preds[0] for p in preds], axis=0)
    rows.append({"slice": "d4_consistency", "bin": "all 8 agree", "n": len(pred),
                 "macro_f1": float("nan"), "defect_recall": float("nan"), "fraction": float(consistent.mean())})

    for rate in (0.005, 0.01, 0.02):
        gen = torch.Generator(device=s.data.device).manual_seed(0)

        def flip(base, rate=rate, gen=gen):
            wafer, fail = base[:, 0], base[:, 1]
            m = (torch.rand(wafer.shape, generator=gen, device=wafer.device) < rate) & (wafer > 0)
            return torch.stack([wafer, torch.where(m, wafer - fail, fail)], 1)
        p = _member_preds(s, flip)
        cm = metrics.confusion(s.y, p)
        rows.append({"slice": "pixel_flip_noise", "bin": f"{rate:.1%}", "n": len(p), "macro_f1": metrics.macro_f1(cm),
                     "defect_recall": metrics.defect_recall_precision(cm)[0]})
    return pd.DataFrame(rows)


def sanity(model_dir: Path) -> dict:
    clf = WaferClassifier(model_dir, "cpu")
    yy, xx = np.mgrid[:52, :52]
    circle = (np.hypot(yy - 25.5, xx - 25.5) <= 26).astype(np.uint8)
    out = {"all_pass": clf.classify(circle)["pattern"], "all_fail": clf.classify(circle * 2)["pattern"]}
    try:
        clf.classify(np.zeros((20, 20), np.uint8))
        out["empty_map"] = "NO ERROR (bad)"
    except ValueError as e:
        out["empty_map"] = f"ValueError: {e}"
    out["all_pass_ok"] = out["all_pass"] == "none"
    out["all_fail_ok"] = out["all_fail"] == "Near-full"
    return out


def figures(s: Scored, out: Path, res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wafer.viz import render_fractions

    cm = np.asarray(res["confusion"])
    for norm in (False, True):
        m = cm / np.maximum(cm.sum(1, keepdims=True), 1) if norm else cm
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        ax.imshow(m, cmap="Blues")
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{m[i, j]:.2f}" if norm else f"{m[i, j]}", ha="center", va="center", fontsize=7,
                        color="white" if m[i, j] > m.max() / 2 else "black")
        ax.set_title("Confusion (row-normalised: recall)" if norm else "Confusion (counts)")
        fig.tight_layout()
        fig.savefig(out / ("confusion_norm.png" if norm else "confusion_counts.png"), dpi=130)
        plt.close(fig)

    conf, correct = s.d["confidence"], s.d["pred"] == s.y
    pdm = s.d["pred"] != NONE_IDX
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, mask, title in ((axes[0], np.ones_like(pdm), "all wafers"), (axes[1], pdm, "predicted defects")):
        c, k = conf[mask], correct[mask]
        order = np.argsort(c)
        bins = np.array_split(order, min(15, max(len(c), 1)))
        ax.plot([0, 1], [0, 1], ls="--", c="grey")
        ax.plot([c[b].mean() for b in bins if len(b)], [k[b].mean() for b in bins if len(b)], marker="o")
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_title(f"Reliability after calibration ({title})")
    fig.tight_layout()
    fig.savefig(out / "reliability_after.png", dpi=130)
    plt.close(fig)

    cr = metrics.coverage_risk(conf[pdm], s.y[pdm], s.d["pred"][pdm])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(cr["coverage"], cr["accuracy"], marker="o", label="accuracy")
    ax.plot(cr["coverage"], cr["macro_f1"], marker="s", label="macro-F1")
    ax.set_xlabel("coverage of predicted-defect wafers (most confident first)")
    ax.set_title("Coverage-risk")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "coverage_risk.png", dpi=130)
    plt.close(fig)

    rr = res["review_rate_per_class"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(rr), [0 if v != v else v for v in rr.values()])
    ax.set_ylabel("share routed to review")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out / "review_rate_per_class.png", dpi=130)
    plt.close(fig)

    wrong = np.flatnonzero(~correct)
    wrong = wrong[np.argsort(-conf[wrong])][:30]
    if len(wrong):
        x = s.data.x[torch.as_tensor(s.idx[wrong], device=s.data.device)].cpu().numpy()
        cols = 6
        fig, axes = plt.subplots(int(np.ceil(len(wrong) / cols)), cols, figsize=(cols * 2, 2.3 * np.ceil(len(wrong) / cols)))
        for ax in np.ravel(axes):
            ax.axis("off")
        for ax, k, xi in zip(np.ravel(axes), wrong, x):
            img = render_fractions(xi[0] / 255, xi[1] / 255) if xi.ndim == 3 else \
                render_fractions((xi > 0).astype(float), (xi == 2).astype(float))
            ax.imshow(img)
            ax.set_title(f"{CLASSES[s.y[k]]} / {CLASSES[s.d['pred'][k]]}\n{conf[k]:.2f}", fontsize=7)
        fig.suptitle("Most confident errors: true / predicted, confidence")
        fig.tight_layout()
        fig.savefig(out / "error_gallery.png", dpi=130)
        plt.close(fig)


def run(model_dir: Path, data_dir: Path, final: bool, comparators: list[Path], out: Path | None,
        device: str, n_boot: int) -> Path:
    split = "test" if final else "val"
    if final:
        out = FINAL_DIR
        if (out / "metrics.json").exists():
            raise SystemExit(f"{out}/metrics.json exists: the test split has already been scored once (Rule 0.1).")
    out = out or REPO_ROOT / "docs" / "results" / f"{model_dir.name}_{split}"
    out.mkdir(parents=True, exist_ok=True)
    s = Scored(model_dir, data_dir, split, device)
    res = metrics.evaluate_predictions(s.y, s.P, s.d, groups=s.groups, n_boot=n_boot)
    res.update({"model": str(model_dir), "split": split, "env": env_info(),
                "note": "validation (calibrated on this set: optimistic)" if not final else "test, scored once"})
    res["sanity_inputs"] = sanity(model_dir)
    sl = slices(s)
    sl.to_csv(out / "slices.csv", index=False)
    pd.DataFrame(res["per_class"]).to_csv(out / "per_class.csv", index=False)

    comps = {}
    for c in comparators:
        sc = Scored(c, data_dir, split, device)
        if not np.array_equal(sc.idx, s.idx):
            raise ValueError(f"{c} scored different rows")
        a_pred, b_pred = s.d["pred"], sc.d["pred"]
        y = s.y
        comps[str(c)] = {
            "macro_f1": metrics.macro_f1(metrics.confusion(y, b_pred)),
            "delta_macro_f1": metrics.paired_bootstrap(
                lambda i: metrics.macro_f1(metrics.confusion(y[i], a_pred[i])),
                lambda i: metrics.macro_f1(metrics.confusion(y[i], b_pred[i])), s.groups, n_boot or 1000),
            "delta_defect_recall": metrics.paired_bootstrap(
                lambda i: metrics.defect_recall_precision(metrics.confusion(y[i], a_pred[i]))[0],
                lambda i: metrics.defect_recall_precision(metrics.confusion(y[i], b_pred[i]))[0], s.groups, n_boot or 1000),
            "mcnemar_secondary": metrics.mcnemar(y, a_pred, b_pred),
        }
    res["comparators"] = comps
    (out / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    figures(s, out, res)
    log.info("%s on %s: macro-F1 %.4f (CI %s), defect recall %.4f -> %s", model_dir.name, split, res["macro_f1"],
             res.get("ci", {}).get("macro_f1"), res["defect_recall"], out)
    return out


def main(argv=None) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="model folder, e.g. models/v0")
    p.add_argument("--final", action="store_true", help="score the TEST split (once, ever)")
    p.add_argument("--comparators", nargs="*", default=[], help="model folders to compare against")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--out", default=None)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return run(Path(a.model), Path(a.data_dir), a.final, [Path(c) for c in a.comparators],
               Path(a.out) if a.out else None, device, a.n_boot)


if __name__ == "__main__":
    main()

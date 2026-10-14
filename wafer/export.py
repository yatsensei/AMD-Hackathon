"""Bundle trained runs into a servable model folder (spec 9.4).

    python -m wafer.export --runs experiments/<run> --version v0
    python -m wafer.export --runs experiments/<a> experiments/<b> experiments/<c> --version v1 --tta

Recomputes the members' validation outputs exactly as `wafer.infer` will (same combine step,
optional D4 TTA), fits calibration on them, and writes models/<version>/.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import yaml

from wafer.env import configure_rocm_env

configure_rocm_env()

import torch  # noqa: E402

from wafer import calibrate, metrics  # noqa: E402
from wafer.bench import latency_ms  # noqa: E402
from wafer.constants import CLASSES  # noqa: E402
from wafer.data.dataset import WaferData  # noqa: E402
from wafer.infer import WaferClassifier, load_members  # noqa: E402
from wafer.predict import predict_rows  # noqa: E402

log = logging.getLogger("wafer.export")
MEMBER_KEYS = ("model", "encoding", "size", "aux_features")


def export(runs: list[Path], version: str, out_root: Path, data_dir: Path, tta: bool, device: str) -> Path:
    cfgs = [yaml.safe_load((r / "config.yaml").read_text()) for r in runs]
    for k in ("encoding", "size", "aux_features"):
        if len({str(c.get(k)) for c in cfgs}) > 1:
            raise ValueError(f"members disagree on {k}: {[c.get(k) for c in cfgs]}")
    out = out_root / version
    if out.exists():
        raise FileExistsError(f"{out} exists; bump the version (spec 9.4) or delete it deliberately")
    (out / "weights").mkdir(parents=True)
    for i, r in enumerate(runs):
        shutil.copy(r / "weights.safetensors", out / "weights" / f"m{i}.safetensors")
    # Team plan: torch.save(model.state_dict(), "cnn.pt"). Same weights as m0; wafer.infer
    # loads the safetensors copies (no pickle), cnn.pt is for anyone loading it by hand.
    from safetensors.torch import load_file
    torch.save(load_file(str(out / "weights" / "m0.safetensors")), out / "cnn.pt")
    c0 = cfgs[0]
    cfg = {"version": version, "encoding": c0["encoding"], "size": int(c0["size"]), "tta": tta,
           "aux_features": bool(c0.get("aux_features", False)),
           "members": [{**{k: c[k] for k in MEMBER_KEYS if k in c}, "source_run": r.name, "seed": c.get("seed")}
                       for c, r in zip(cfgs, runs)]}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "classes.json").write_text(json.dumps(CLASSES))

    data = WaferData(data_dir / f"wafers_s{cfg['size']}.npz", data_dir / "metadata.parquet",
                     encoding=cfg["encoding"], device=device, aux=cfg["aux_features"])
    va = data.indices("val")
    members = load_members(out, cfg, device)
    z = calibrate.combine_members([predict_rows(m, data, va, tta=tta) for m in members], tta)
    priors = [json.loads((r / "train_prior.json").read_text()) for r in runs]
    pi_nat = priors[0]["pi_natural"]
    pi_eff = np.mean([p["pi_train_eff"] for p in priors], axis=0)
    rc = c0.get("recall_target", calibrate.DEFAULT_RECALL_TARGET)
    sa = c0.get("sel_acc_target", calibrate.DEFAULT_SEL_ACC_TARGET)
    y = data.y_np[va]
    cal = calibrate.calibrate(z, y, pi_nat, pi_eff, rc, sa)
    (out / "calibration.json").write_text(json.dumps(cal, indent=2))
    P = calibrate.apply(cal, z)
    groups = data.meta["split_group"].to_numpy()[va] if "split_group" in data.meta else None
    val = metrics.evaluate_predictions(y, P, calibrate.decide(cal, P), groups=groups, n_boot=1000)
    val["note"] = "validation; calibration and thresholds were fitted on this same set (optimistic)"
    (out / "metrics_val.json").write_text(json.dumps(val, indent=2))

    clf = WaferClassifier(out, "cpu")
    ms = latency_ms(clf.classify)
    import platform
    ci = val.get("ci", {}).get("macro_f1", {})
    (out / "README.md").write_text(f"""# Model {version}

- Members: {len(runs)} × {', '.join(sorted({c['model'] for c in cfgs}))} · encoding `{cfg['encoding']}` · {cfg['size']}px · TTA: {'D4 ×8' if tta else 'off'}
- Source runs: {', '.join(r.name for r in runs)}
- Calibration (fitted on val): T = {cal['temperature']:.3f} · θ_none = {cal['theta_none']:.4f} (defect recall target {rc}) · τ = {cal['tau']:.4f} (selective accuracy target {sa}, defect coverage {cal['val_defect_coverage_at_tau']:.3f})
- **CPU latency (served config, `classify`, batch 1, 4 threads): {ms:.1f} ms** on `{platform.processor() or platform.machine()}` — budget 50 ms (spec 9.1)

## Validation (optimistic: calibrated on the same set)
| metric | value |
|---|---|
| macro-F1 (9) | {val['macro_f1']:.4f} (95% CI {ci.get('lo', float('nan')):.4f}–{ci.get('hi', float('nan')):.4f}) |
| defect macro-F1 (8) | {val['defect_macro_f1']:.4f} |
| balanced accuracy | {val['balanced_accuracy']:.4f} |
| defect recall / precision | {val['defect_recall']:.4f} / {val['defect_precision']:.4f} |
| ECE on predicted defects | {val['ece_defect']:.4f} |
| review rate | {val['review_rate']:.4f} |

Never quote these as test numbers. Test is scored once, by `python -m wafer.evaluate --final`.
""")
    log.info("exported %s: val macro-F1 %.4f, CPU %.1f ms/wafer", out, val["macro_f1"], ms)
    return out


def main(argv=None) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="experiment run folders (members)")
    p.add_argument("--version", required=True, help="e.g. v0, v1")
    p.add_argument("--out", default="models")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--tta", action="store_true", help="serve with D4 test-time augmentation (x8 cost)")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return export([Path(r) for r in a.runs], a.version, Path(a.out), Path(a.data_dir), a.tta, device)


if __name__ == "__main__":
    main()

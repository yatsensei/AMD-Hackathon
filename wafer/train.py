"""Train one config (spec 5.6). Every run is reproducible from its config.

    python -m wafer.train --config configs/e0_baseline.yaml              # every seed in the config
    python -m wafer.train --config configs/e0_baseline.yaml --seed 3     # one seed (parallel sweeps)

Writes experiments/<run_id>/: config.yaml, weights.safetensors, history.csv, val_logits.npy,
val_uids.npy, train_prior.json, calibration.json, val_metrics.json, summary.json, env.json.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from wafer.env import configure_rocm_env, env_info

configure_rocm_env()

import torch  # noqa: E402

from wafer import calibrate, metrics  # noqa: E402
from wafer.constants import DEFECT_IDX  # noqa: E402
from wafer.data import sampler  # noqa: E402
from wafer.data.augment import augment  # noqa: E402
from wafer.data.dataset import WaferData, to_input  # noqa: E402
from wafer.models.factory import build_model  # noqa: E402
from wafer.losses import make_loss  # noqa: E402
from wafer.predict import predict_rows  # noqa: E402

log = logging.getLogger("wafer.train")

DEFAULTS = {
    "encoding": "area", "size": 64, "loss": "weighted_ce", "sampler": "none_downsample",
    "none_per_epoch": 15000, "aug": {"d4": True, "small_rot": False, "die_flip": 0.0},
    "optim": {"lr": 1.0e-3, "wd": 1.0e-4, "epochs": 30, "batch_size": 256, "warmup_epochs": 1},
    "seeds": [0], "aux_features": False, "amp": True, "compile": False,
    "recall_target": calibrate.DEFAULT_RECALL_TARGET, "sel_acc_target": calibrate.DEFAULT_SEL_ACC_TARGET,
}


def load_config(path: str | Path) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    user = yaml.safe_load(Path(path).read_text()) or {}
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    cfg["config_name"] = Path(path).stem
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_lambda(warmup_steps: int, total_steps: int):
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return f


def cpu_latency_ms(model, cfg: dict, n: int = 50, warmup: int = 10, threads: int = 4) -> float:
    """Single model, batch 1, no TTA, on CPU (reference setup in spec 9.1)."""
    m = copy.deepcopy(model).float().cpu().eval()
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        from wafer.models.factory import ENCODING_CHANNELS
        x = torch.rand(1, ENCODING_CHANNELS[cfg["encoding"]], cfg["size"], cfg["size"])
        aux = torch.zeros(1, 3) if cfg.get("aux_features") else None
        with torch.no_grad():
            for _ in range(warmup):
                m(x, aux)
            t = time.perf_counter()
            for _ in range(n):
                m(x, aux)
        return (time.perf_counter() - t) / n * 1000
    finally:
        torch.set_num_threads(prev)


def save_weights(model, path: Path) -> None:
    from safetensors.torch import save_file
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(path))


def train_one(cfg: dict, seed: int, data_dir: Path, out_root: Path, device: str,
              max_steps: int | None = None) -> Path:
    t_start = time.time()
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = seed
    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    run_id = f"{cfg['config_name']}_s{seed}_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out = out_root / run_id
    out.mkdir(parents=True, exist_ok=False)
    log.info("run %s on %s", run_id, device)

    data = WaferData(data_dir / f"wafers_s{cfg['size']}.npz", data_dir / "metadata.parquet",
                     encoding=cfg["encoding"], device=device, aux=cfg["aux_features"])
    tr, va = data.indices("train"), data.indices("val")
    y_tr = data.y_np[tr]
    counts = sampler.class_counts(y_tr)
    w = sampler.class_weights(counts, cfg["loss"])
    pi_nat = counts / counts.sum()
    pi_eff = sampler.effective_prior(counts, cfg["sampler"], cfg["none_per_epoch"], cfg["loss"])
    (out / "train_prior.json").write_text(json.dumps({
        "pi_natural": pi_nat.tolist(), "pi_train_eff": pi_eff.tolist(), "train_counts": counts.tolist(),
        "class_weights": w.tolist(), "epoch_counts": sampler.epoch_counts(counts, cfg["sampler"],
                                                                          cfg["none_per_epoch"]).tolist()}, indent=2))

    model = build_model(cfg).to(device)
    fwd = torch.compile(model) if cfg.get("compile") else model
    crit = make_loss(cfg["loss"], torch.tensor(w, dtype=torch.float32, device=device))
    o = cfg["optim"]
    opt = torch.optim.AdamW(model.parameters(), lr=o["lr"], weight_decay=o["wd"])
    rng = np.random.default_rng(seed)
    n_epoch = len(sampler.epoch_indices(tr, y_tr, cfg["sampler"], cfg["none_per_epoch"], np.random.default_rng(0)))
    steps_per_epoch = math.ceil(n_epoch / o["batch_size"])
    total = o["epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(o["warmup_epochs"] * steps_per_epoch, total))
    amp = bool(cfg["amp"]) and device.startswith("cuda")

    history, step = [], 0
    for epoch in range(o["epochs"]):
        model.train()
        t_ep = time.time()
        idx = sampler.epoch_indices(tr, y_tr, cfg["sampler"], cfg["none_per_epoch"], rng)
        loss_sum, n_seen = 0.0, 0
        for i in range(0, len(idx), o["batch_size"]):
            base, y, aux = data.batch(idx[i:i + o["batch_size"]])
            x = to_input(augment(base, cfg["aug"]), cfg["encoding"])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = fwd(x, aux)
            loss = crit(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(y)
            n_seen += len(y)
            step += 1
            if max_steps and step >= max_steps:
                break
        z_val = predict_rows(model, data, va, amp=amp)
        y_val = data.y_np[va]
        cm = metrics.confusion(y_val, z_val.argmax(1))
        row = {"epoch": epoch + 1, "train_loss": loss_sum / max(n_seen, 1),
               "val_loss": metrics.nll(y_val, calibrate.softmax(z_val)),
               "val_macro_f1": metrics.macro_f1(cm), "val_defect_macro_f1": metrics.macro_f1(cm, DEFECT_IDX),
               "val_bal_acc": metrics.balanced_accuracy(cm), "lr": sched.get_last_lr()[0],
               "seconds": time.time() - t_ep}
        history.append(row)
        log.info("epoch %d/%d loss %.4f val_loss %.4f val_macro_f1 %.4f (%.1fs)", row["epoch"], o["epochs"],
                 row["train_loss"], row["val_loss"], row["val_macro_f1"], row["seconds"])
        if max_steps and step >= max_steps:
            break

    # Final checkpoint only (no early stopping, spec 5.6).
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    save_weights(model, out / "weights.safetensors")
    z_val = predict_rows(model, data, va, amp=False)
    y_val = data.y_np[va]
    np.save(out / "val_logits.npy", z_val.astype(np.float32))
    np.save(out / "val_uids.npy", data.uid[va])

    cal = calibrate.calibrate(z_val, y_val, pi_nat, pi_eff, cfg["recall_target"], cfg["sel_acc_target"])
    (out / "calibration.json").write_text(json.dumps(cal, indent=2))
    P = calibrate.apply(cal, z_val)
    val = metrics.evaluate_predictions(y_val, P, calibrate.decide(cal, P))
    val["note"] = "calibrated and thresholded on this same val set: optimistic (spec 7.4)"
    (out / "val_metrics.json").write_text(json.dumps(val, indent=2))

    cfg_out = {k: v for k, v in cfg.items() if k != "seeds"}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg_out, sort_keys=False))
    env = env_info()
    (out / "env.json").write_text(json.dumps(env, indent=2))
    aug = cfg["aug"]
    summary = {
        "run_id": run_id, "experiment": cfg.get("experiment", ""), "config": cfg["config_name"],
        "model": cfg["model"], "encoding": cfg["encoding"], "loss": cfg["loss"], "sampler": cfg["sampler"],
        "aug": f"d4={int(aug.get('d4', True))},rot={int(aug.get('small_rot', False))},flip={aug.get('die_flip', 0)}",
        "seed": seed, "val_macro_f1": val["macro_f1"], "val_defect_macro_f1": val["defect_macro_f1"],
        "val_bal_acc": val["balanced_accuracy"], "val_defect_recall": val["defect_recall"],
        "val_defect_precision": val["defect_precision"], "val_macro_f1_argmax_raw": metrics.macro_f1(
            metrics.confusion(y_val, z_val.argmax(1))),
        "val_ece_defect": val["ece_defect"], "val_review_rate": val["review_rate"],
        "cpu_ms": cpu_latency_ms(model, cfg), "epochs": len(history),
        "train_minutes": (time.time() - t_start) / 60, "gpu": env.get("gpu_name", "cpu"),
        "git_commit": env["git_commit"], "dirty": env["git_dirty"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("done %s: val macro-F1 %.4f (operating point), %.4f (raw argmax); %.1f min",
             run_id, summary["val_macro_f1"], summary["val_macro_f1_argmax_raw"], summary["train_minutes"])
    return out


def main(argv=None) -> list[Path]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None, help="run only this seed")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--out", default="experiments")
    p.add_argument("--device", default=None, help="default: cuda if available (ROCm GPU), else cpu")
    p.add_argument("--epochs", type=int, default=None, help="override optim.epochs")
    p.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests only)")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    if a.epochs:
        cfg["optim"]["epochs"] = a.epochs
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [a.seed] if a.seed is not None else list(cfg["seeds"])
    return [train_one(cfg, s, Path(a.data_dir), Path(a.out), device, a.max_steps) for s in seeds]


if __name__ == "__main__":
    main()

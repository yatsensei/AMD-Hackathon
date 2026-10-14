"""Phase 1 end to end: LSWMD.pkl -> wafers_s64.npz + metadata.parquet + data report (spec 3).

    python scripts/prepare_data.py                      # critical path (Thu 10:00)
    python scripts/prepare_data.py --sizes 64 96 --yolo # plus non-critical outputs
"""
from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from wafer.constants import CLASSES
from wafer.data import preprocess, split as splitlib
from wafer.data.load import load_raw, sha256_file
from wafer.data.report import recorded_sha256, write_report

log = logging.getLogger("wafer.prepare_data")

META_COLUMNS = ["uid", "lotName", "split_group", "waferIndex", "label", "label_idx", "split", "cv_fold",
                "orig_h", "orig_w", "n_die", "n_fail", "dieSize", "hash", "dup_group", "orig_split"]


def _parse_min_count(items: list[str]) -> dict:
    mc = dict(splitlib.DEFAULT_MIN_COUNT)
    for item in items or []:
        cls, _, n = item.partition("=")
        if cls not in CLASSES or not n.isdigit():
            raise SystemExit(f"--min-count expects CLASS=N with CLASS in {CLASSES}, got {item!r}")
        mc[cls] = int(n)
    return mc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="data/raw/LSWMD.pkl")
    p.add_argument("--interim-dir", default="data/interim")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--report", default="docs/results/data_report.md")
    p.add_argument("--expected-sha256", default=None,
                   help="defaults to the hash recorded in --report; first run records it")
    p.add_argument("--sizes", type=int, nargs="+", default=[64])
    p.add_argument("--split-seeds", type=int, default=100, help="try seeds 0..N-1")
    p.add_argument("--min-count", action="append", metavar="CLASS=N",
                   help=f"override minimum val/test count per class (default {splitlib.DEFAULT_MIN_COUNT})")
    p.add_argument("--max-rel-dev", type=float, default=splitlib.DEFAULT_MAX_REL_DEV)
    p.add_argument("--cv-seed", type=int, default=0)
    p.add_argument("--dup-min-minority", type=int, default=preprocess.DUP_MIN_MINORITY,
                   help="duplicate maps link lots only if min(n_fail, n_pass) >= this")
    p.add_argument("--pin-train-frac", type=float, default=splitlib.DEFAULT_PIN_TRAIN_FRAC,
                   help="linked groups holding more than this share of wafers always go to train")
    p.add_argument("--jobs", type=int, default=None, help="processes for the split search")
    p.add_argument("--yolo", action="store_true", help="also write yolo_data/ PNGs (non-critical)")
    p.add_argument("--yolo-dir", default="yolo_data")
    p.add_argument("--yolo-encoding", choices=["area", "nearest"], default="area")
    p.add_argument("--yolo-size", type=int, default=64)
    return p


def export_yolo(meta: pd.DataFrame, x_area: np.ndarray, x_near: np.ndarray, out_dir: Path,
                encoding: str) -> int:
    from PIL import Image

    from wafer.viz import render_fractions, render_wafer
    for s in meta["split"].unique():
        for c in CLASSES:
            (out_dir / s / c).mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(meta[["uid", "split", "label"]].itertuples(index=False)):
        if encoding == "area":
            img = render_fractions(x_area[i, 0] / 255.0, x_area[i, 1] / 255.0)
        else:
            img = render_wafer(x_near[i])
        Image.fromarray(img).save(out_dir / row.split / row.label / f"{row.uid}.png")
    return len(meta)


def run(args: argparse.Namespace, command: str = "") -> pd.DataFrame:
    t0 = time.time()
    raw_path = Path(args.raw)
    interim, out = Path(args.interim_dir), Path(args.out_dir)
    interim.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    min_count = _parse_min_count(args.min_count)

    expected = args.expected_sha256 or recorded_sha256(args.report)
    if expected:
        raw = load_raw(raw_path, expected_sha256=expected)   # raises before unpickling on mismatch
        sha = expected.lower()
    else:
        sha = sha256_file(raw_path)
        log.warning("no SHA-256 on record: recording %s in %s; commit the report", sha, args.report)
        raw = load_raw(raw_path)
    log.info("loaded %d raw rows (%.0fs)", len(raw), time.time() - t0)

    lab, unl_meta, info = preprocess.clean(raw)
    del raw
    log.info("clean: %d labelled, %d unlabelled, dropped %d invalid + %d empty, %d lots filled",
             info.n_labeled, info.n_unlabeled, info.n_dropped_invalid, info.n_dropped_empty,
             info.n_lot_filled)
    lab.to_pickle(interim / "labeled.pkl")
    unl_meta.to_parquet(interim / "unlabeled_meta.parquet", index=False)

    lab["dup_group"] = preprocess.assign_dup_groups(lab)
    dup = preprocess.duplicate_report(lab, args.dup_min_minority)
    lab["split_group"], link = preprocess.link_lots(lab, args.dup_min_minority)
    log.info("duplicates: %d groups (%d informative, %d with conflicting labels); "
             "%d lots linked into shared split groups, largest group %d wafers / %d lots",
             dup.n_groups, dup.n_informative_groups, dup.n_conflicting_groups, link.n_lots_linked,
             link.largest_group_wafers, link.largest_group_lots)
    pin = splitlib.oversized_groups(lab["split_group"].to_numpy(), args.pin_train_frac)
    if pin.any():
        log.warning("%d wafers in linked groups larger than %.1f%% of the data are pinned to train "
                    "(largest: %d wafers / %d lots); see the duplicates section of the report",
                    int(pin.sum()), 100 * args.pin_train_frac, link.largest_group_wafers, link.largest_group_lots)

    log.info("split search over %d seeds", args.split_seeds)
    search = splitlib.search_split(lab["label"].to_numpy(), lab["split_group"].to_numpy(),
                                   seeds=range(args.split_seeds), min_count=min_count, jobs=args.jobs,
                                   pin_train=pin)
    lab["split"] = search.split
    lab = lab.sort_values("uid").reset_index(drop=True)
    lab["cv_fold"] = splitlib.assign_cv_folds(lab["label"].to_numpy(), lab["split_group"].to_numpy(),
                                              lab["split"].to_numpy(), seed=args.cv_seed)
    rel_dev = splitlib.check_split(lab, min_count=min_count, max_rel_dev=args.max_rel_dev)

    meta = lab[META_COLUMNS].copy()
    meta["split"] = meta["split"].astype(str)
    meta.to_parquet(out / "metadata.parquet", index=False)
    (out / "split_params.json").write_text(json.dumps({
        "seed": search.seed, "seeds_tried": args.split_seeds, "n_feasible": search.n_feasible,
        "worst_rel_dev": search.chosen.worst_dev, "min_count": min_count,
        "max_rel_dev": args.max_rel_dev, "cv_seed": args.cv_seed,
        "dup_min_minority": args.dup_min_minority, "pin_train_frac": args.pin_train_frac,
        "n_pinned_to_train": int(pin.sum())}, indent=2))

    encoded = {}
    for s in args.sizes:
        t = time.time()
        x_area, x_near = preprocess.encode_all(lab["waferMap"].tolist(), size=s)
        np.savez_compressed(out / f"wafers_s{s}.npz", X_area=x_area, X_nearest=x_near,
                            y=lab["label_idx"].to_numpy(np.int64), uid=lab["uid"].to_numpy(np.int64))
        if args.yolo and s == args.yolo_size:
            encoded[s] = (x_area, x_near)
        log.info("wrote wafers_s%d.npz (%.0fs)", s, time.time() - t)

    write_report(args.report, sha256=sha, sha_verified=bool(expected), clean_info=info,
                 dup_report=dup, link_report=link, dup_min_minority=args.dup_min_minority,
                 n_pinned=int(pin.sum()), pin_train_frac=args.pin_train_frac,
                 search=search, min_count=min_count, max_rel_dev=args.max_rel_dev, rel_dev=rel_dev,
                 cv_seed=args.cv_seed, meta=meta, sizes=args.sizes,
                 command=command)
    log.info("wrote %s", args.report)

    if args.yolo:
        s = args.yolo_size
        x_area, x_near = encoded.get(s) or preprocess.encode_all(lab["waferMap"].tolist(), size=s)
        n = export_yolo(meta, x_area, x_near, Path(args.yolo_dir), args.yolo_encoding)
        log.info("wrote %d PNGs to %s (%s encoding)", n, args.yolo_dir, args.yolo_encoding)

    log.info("done in %.0fs", time.time() - t0)
    return meta


def main(argv: list[str] | None = None) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)
    return run(args, command=" ".join(["python scripts/prepare_data.py", *map(shlex.quote, argv)]))


if __name__ == "__main__":
    main()

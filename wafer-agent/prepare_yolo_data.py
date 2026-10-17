"""
Turn WM-811K wafer maps into a YOLO *detection* dataset.

WM-811K has one label per wafer and no bounding boxes, so we generate boxes
automatically: find the clusters of failed chips on each map and draw a box
around the big ones. Each box gets the wafer's label (Center, Scratch, ...).
This is "weak labelling" -- good enough to train YOLO to say WHAT the defect
is and WHERE it sits on the wafer. Say this honestly in the pitch.

Usage:
    python prepare_yolo_data.py --pkl LSWMD.pkl --out yolo_wafer --per-class 3000
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# "none" wafers are kept as background images (no boxes), not as a class.
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch"]
WHOLE_WAFER = {"Near-full", "Random"}  # defect covers the whole wafer -> one big box
SINGLE_DEFECT = {"Center", "Loc", "Edge-Loc", "Scratch"}  # one main cluster -> box only the biggest

# Colours for the rendered image (BGR): no chip / pass / fail
COLORS = {0: (0, 0, 0), 1: (90, 140, 60), 2: (40, 220, 255)}


def load_wm811k(pkl_path):
    # The pickle was saved with an old pandas; this alias lets new pandas read it.
    import pandas.core.indexes
    sys.modules["pandas.indexes"] = pandas.core.indexes
    try:
        df = pd.read_pickle(pkl_path)
    except UnicodeDecodeError:
        # The file was written by Python 2; read its text as latin-1.
        try:
            from pandas.compat.pickle_compat import Unpickler
            with open(pkl_path, "rb") as f:
                df = Unpickler(f, encoding="latin1").load()
        except Exception as e:
            sys.exit(
                f"Could not read {pkl_path} with pandas {pd.__version__} ({e}).\n"
                'Fix: run  pip install "pandas<3"  and try again.'
            )

    def flat(x):
        try:
            return str(np.asarray(x).ravel()[0]) if np.asarray(x).size > 0 else None
        except Exception:
            return None

    df["label"] = df["failureType"].apply(flat)
    df = df[df["label"].notna()].reset_index(drop=True)
    return df[["waferMap", "lotName", "waferIndex", "label"]]


def render(wmap, size):
    wmap = np.asarray(wmap, dtype=np.uint8)
    small = cv2.resize(wmap, (size, size), interpolation=cv2.INTER_NEAREST)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for v, c in COLORS.items():
        img[small == v] = c
    return small, img


def make_boxes(small, label, keep_frac=0.3, min_area_frac=0.002):
    """Return YOLO boxes [(cx, cy, w, h)] normalised 0-1."""
    size = small.shape[0]
    if label in WHOLE_WAFER:
        ys, xs = np.where(small > 0)  # whole die area
        if len(xs) == 0:
            return []
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        return [((x0 + x1) / 2 / size, (y0 + y1) / 2 / size, (x1 - x0) / size, (y1 - y0) / size)]

    fail = (small == 2).astype(np.uint8)
    # Dilate so nearby failed chips join into one cluster. Scratches are often broken
    # into segments, so they get a bigger dilation to join the pieces into one line.
    k = 7 if label == "Scratch" else 3
    fail = cv2.dilate(fail, np.ones((k, k), np.uint8), iterations=1)
    n, _, stats, _ = cv2.connectedComponentsWithStats(fail, connectivity=8)
    if n <= 1:
        return []
    comps = stats[1:]  # drop background
    areas = comps[:, cv2.CC_STAT_AREA]

    if label in SINGLE_DEFECT:
        # One defect per wafer: box only the biggest cluster, ignore background specks.
        x, y, w, h, a = comps[areas.argmax()]
        if a < size * size * min_area_frac:
            return []
        return [((x + w / 2) / size, (y + h / 2) / size, w / size, h / size)]

    # Edge-Ring / Donut: the ring can come out as several arcs, keep the big ones.
    min_area = max(areas.max() * keep_frac, size * size * min_area_frac)
    boxes = []
    for x, y, w, h, a in comps:
        if a >= min_area:
            boxes.append(((x + w / 2) / size, (y + h / 2) / size, w / size, h / size))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="LSWMD.pkl")
    ap.add_argument("--out", default="yolo_wafer")
    ap.add_argument("--size", type=int, default=128, help="image size in pixels")
    ap.add_argument("--per-class", type=int, default=3000, help="max wafers per defect class")
    ap.add_argument("--none", type=int, default=3000, help="number of 'none' background wafers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["cls", "detect"], default="cls",
                    help="cls = YOLO classification (recommended, no boxes needed); "
                         "detect = YOLO detection with auto-generated boxes")
    args = ap.parse_args()

    random.seed(args.seed)
    df = load_wm811k(args.pkl)
    print("Labelled wafers per class:\n", df["label"].value_counts().to_string())

    # Balance: cap each class, add some 'none' as negatives.
    parts = []
    for lab, g in df.groupby("label"):
        cap = args.none if lab == "none" else args.per_class
        parts.append(g.sample(min(len(g), cap), random_state=args.seed))
    df = pd.concat(parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # 70 / 15 / 15 split, stratified by label
    df["split"] = "train"
    for lab, g in df.groupby("label"):
        idx = g.index.tolist()
        n = len(idx)
        df.loc[idx[: int(0.15 * n)], "split"] = "val"
        df.loc[idx[int(0.15 * n): int(0.30 * n)], "split"] = "test"

    out = Path(args.out)

    if args.mode == "cls":
        # YOLO classification layout: <out>/<split>/<class>/<image>.png  ("none" is a class here)
        rows = []
        for i, r in df.iterrows():
            _, img = render(r["waferMap"], args.size)
            d = out / r["split"] / r["label"]
            d.mkdir(parents=True, exist_ok=True)
            name = f"w{i:06d}"
            cv2.imwrite(str(d / f"{name}.png"), img)
            rows.append({"image": name, "split": r["split"], "label": r["label"],
                         "lotName": r["lotName"], "waferIndex": r["waferIndex"]})
        pd.DataFrame(rows).to_csv(out / "index.csv", index=False)
        counts = pd.DataFrame(rows).groupby(["split"]).size().to_dict()
        print(f"Done: {len(rows)} images in {out}/ (per split: {counts}). "
              f"Train with: yolo classify train data={out} ...")
        return

    for s in ("train", "val", "test"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    rows, n_boxes, skipped = [], 0, 0
    for i, r in df.iterrows():
        small, img = render(r["waferMap"], args.size)
        lab = r["label"]
        boxes = [] if lab == "none" else make_boxes(small, lab)
        if lab != "none" and not boxes:
            skipped += 1  # defect wafer with no visible cluster -> drop it
            continue
        name = f"w{i:06d}"
        cv2.imwrite(str(out / "images" / r["split"] / f"{name}.png"), img)
        with open(out / "labels" / r["split"] / f"{name}.txt", "w") as f:
            for cx, cy, w, h in boxes:
                f.write(f"{CLASSES.index(lab)} {cx:.5f} {cy:.5f} {w:.5f} {h:.5f}\n")
        n_boxes += len(boxes)
        rows.append({"image": name, "split": r["split"], "label": lab,
                     "lotName": r["lotName"], "waferIndex": r["waferIndex"]})

    pd.DataFrame(rows).to_csv(out / "index.csv", index=False)  # keeps lot info for the simulation
    (out / "wafer.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        + "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    print(f"Done: {len(rows)} images, {n_boxes} boxes, {skipped} skipped. Dataset config: {out/'wafer.yaml'}")


if __name__ == "__main__":
    main()
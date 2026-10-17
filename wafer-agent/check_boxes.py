"""
Draw the generated boxes on a few real wafers per class, and count boxes per class.

    python check_boxes.py --data yolo_wafer
Opens/saves box_check_real.png -- one row per class, 6 wafers each.
"""
import argparse
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="yolo_wafer")
ap.add_argument("--n", type=int, default=6)
args = ap.parse_args()
d = Path(args.data)
idx = pd.read_csv(d / "index.csv")
random.seed(1)

S = 160
rows, stats = [], []
for lab in sorted(idx.label.unique()):
    sub = idx[idx.label == lab]
    counts = [sum(1 for _ in open(d / "labels" / r.split / f"{r.image}.txt")) for r in sub.itertuples()]
    c = Counter(counts)
    stats.append((lab, len(sub), np.mean(counts), c.get(1, 0) / len(sub)))

    tiles = []
    for r in sub.sample(min(args.n, len(sub)), random_state=1).itertuples():
        img = cv2.resize(cv2.imread(str(d / "images" / r.split / f"{r.image}.png")), (S, S),
                         interpolation=cv2.INTER_NEAREST)
        for line in open(d / "labels" / r.split / f"{r.image}.txt"):
            _, cx, cy, w, h = map(float, line.split())
            cv2.rectangle(img, (int((cx - w / 2) * S), int((cy - h / 2) * S)),
                          (int((cx + w / 2) * S) - 1, int((cy + h / 2) * S) - 1), (0, 0, 255), 2)
        tiles.append(img)
    while len(tiles) < args.n:
        tiles.append(np.zeros((S, S, 3), np.uint8))
    label = np.zeros((S, 120, 3), np.uint8)
    cv2.putText(label, lab, (5, S // 2), 0, 0.5, (255, 255, 255), 1)
    rows.append(np.hstack([label] + tiles))

cv2.imwrite("box_check_real.png", np.vstack(rows))
print(f"{'class':<10} {'wafers':>7} {'avg boxes':>10} {'% with 1 box':>13}")
for lab, n, avg, one in stats:
    print(f"{lab:<10} {n:>7} {avg:>10.2f} {one*100:>12.0f}%")
print("Saved box_check_real.png")

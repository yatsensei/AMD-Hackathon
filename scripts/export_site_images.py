"""Render the homepage's wafer images into dashboard/site/img/ (real validation wafers).

    python scripts/export_site_images.py

One map per defect pattern, picked from the validation split where the recommended
model is confident and correct, plus one attention overlay. Needs data/interim/labeled.pkl.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from wafer.constants import CLASSES
from wafer.data.access import get_wafer_map, load_split
from wafer.env import REPO_ROOT
from wafer.infer import load_model

OUT = REPO_ROOT / "dashboard" / "site" / "img"
# Dark-section palette: off-wafer transparent, passing die graphite, failing die red.
DARK = np.array([[0, 0, 0, 0], [46, 48, 54, 255], [226, 35, 45, 255]], dtype=np.uint8)
SCALE = 6


def save(m: np.ndarray, name: str) -> None:
    img = Image.fromarray(DARK[np.asarray(m, dtype=np.intp)])
    img.resize((m.shape[1] * SCALE, m.shape[0] * SCALE), Image.NEAREST).save(OUT / name, optimize=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_model(REPO_ROOT / "models" / "v0-ens3", "cpu")
    split = load_split()
    val = split[(split["split"] == "val") & split["n_die"].between(700, 2600)]
    rng = np.random.default_rng(7)
    picked = {}
    for c in CLASSES:
        rows = val[val["label"] == c]
        ok = []
        for uid in rng.permutation(rows["uid"].to_numpy())[:60]:
            m = get_wafer_map(int(uid))
            r = model.classify(m)
            if r["pattern"] == c and r["confidence"] > 0.97:
                ok.append((int(uid), m))
            if len(ok) == 8:
                break
        # Clearest example: fewest background fails for "none", most visible pattern otherwise.
        share = [float((m == 2).sum()) / max(1, int((m > 0).sum())) for _, m in ok]
        picked[c] = ok[int(np.argmin(share) if c == "none" else np.argmax(share) if c != "Near-full" else 0)]
        if c == "Edge-Ring":                         # three different wafers for the alert preview
            for i, (_, m) in enumerate(ok[:3], 1):
                save(m, f"wafer-edge-ring-{i}.png")
    for c, (uid, m) in picked.items():
        save(m, f"wafer-{c.lower()}.png")
        print(c, uid, m.shape)
    # Attention overlay for the Scratch example: dies tinted by where the model looked.
    uid, m = picked["Scratch"]
    heat = np.asarray(model.explain(m)["heatmap"], dtype=np.float32)
    heat = (heat - heat.min()) / (np.ptp(heat) + 1e-9)
    rgba = DARK[np.asarray(m, dtype=np.intp)].astype(np.float32)
    on = m > 0
    glow = np.array([255, 205, 208], dtype=np.float32)
    a = (0.85 * heat[..., None]) * on[..., None]
    rgba[..., :3] = rgba[..., :3] * (1 - a) + glow * a
    rgba[(m == 2), :3] = DARK[2, :3]
    img = Image.fromarray(rgba.astype(np.uint8))
    img.resize((m.shape[1] * SCALE, m.shape[0] * SCALE), Image.NEAREST).save(OUT / "wafer-scratch-attention.png", optimize=True)


if __name__ == "__main__":
    main()

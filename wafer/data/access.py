"""Read-only access to the dataset for other phases (factory stream, dashboard, agent).

    from wafer.data.access import load_split, get_wafer_map, wafers_for_stream
    split = load_split()                       # committed: splits/wafer_split_v1.csv.gz (no data build needed)
    m = get_wafer_map(uid)                     # raw 2D map (0/1/2)
    df = wafers_for_stream("val")              # lots/wafers to stream; switch to "test" only after the final eval

Maps resolve in this order:
1. `dashboard/data/stream_maps.npz` — 500 val-stream wafers for the demo / Space (no 2 GB dataset)
2. `data/interim/labeled.pkl` — full labelled set from `python scripts/prepare_data.py`

The split file is the source of truth for which wafer is in which split, so every machine
uses the same one.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from wafer.env import REPO_ROOT

SPLIT_FILE = REPO_ROOT / "splits" / "wafer_split_v1.csv.gz"
LABELED = REPO_ROOT / "data" / "interim" / "labeled.pkl"
DEMO_MAPS = REPO_ROOT / "dashboard" / "data" / "stream_maps.npz"


@lru_cache(maxsize=1)
def load_split(path: str | Path = SPLIT_FILE) -> pd.DataFrame:
    """uid, lotName, waferIndex, label, split, split_group, cv_fold, orig_h, orig_w, n_die, n_fail, ..."""
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def load_demo_maps(path: str = str(DEMO_MAPS)) -> dict[int, np.ndarray]:
    """uid → raw map for the bundled validation stream (Space / laptop demo without labeled.pkl)."""
    p = Path(path)
    if not p.exists():
        return {}
    z = np.load(p)
    out, offset = {}, 0
    for uid, (h, w) in zip(z["uid"], z["shape"]):
        out[int(uid)] = z["flat"][offset:offset + h * w].reshape(h, w)
        offset += h * w
    return out


@lru_cache(maxsize=1)
def _maps(path: str = str(LABELED)) -> pd.Series:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: run `python scripts/prepare_data.py` once to build it")
    return pd.read_pickle(p).set_index("uid")["waferMap"]


def get_wafer_map(uid: int) -> np.ndarray:
    """Raw wafer map (uint8, 0 = no die, 1 = pass, 2 = fail) for a uid.

    Prefers the demo stream pack so dashboard and `classify_wafer` work without the raw dataset.
    """
    uid = int(uid)
    demo = load_demo_maps()
    if uid in demo:
        return demo[uid]
    try:
        return _maps().at[uid]
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"no map for uid {uid}: not in {DEMO_MAPS.name} and labeled.pkl missing"
        ) from e
    except KeyError as e:
        raise FileNotFoundError(f"no map for uid {uid}") from e


def wafers_for_stream(split: str = "val") -> pd.DataFrame:
    """Wafers the factory simulation may stream, ordered by lot then wafer index.

    Spec 9.3: use "val" while building the demo; switch to "test" only after the single final
    test evaluation has been run, so nobody looks at test predictions beforehand.
    """
    if split not in ("val", "test"):
        raise ValueError("stream only val or test wafers: the model was trained on train")
    s = load_split()
    return s[s["split"] == split].sort_values(["lotName", "waferIndex"]).reset_index(drop=True)

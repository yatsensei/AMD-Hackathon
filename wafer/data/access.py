"""Read-only access to the dataset for other phases (factory stream, dashboard, agent).

    from wafer.data.access import load_split, get_wafer_map, wafers_for_stream
    split = load_split()                       # committed: splits/wafer_split_v1.csv.gz (no data build needed)
    m = get_wafer_map(uid)                     # raw 2D map (0/1/2); needs data/interim/labeled.pkl
    df = wafers_for_stream("val")              # lots/wafers to stream; switch to "test" only after the final eval

Maps come from `data/interim/labeled.pkl`, written by `python scripts/prepare_data.py` (~7 min).
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


@lru_cache(maxsize=1)
def load_split(path: str | Path = SPLIT_FILE) -> pd.DataFrame:
    """uid, lotName, waferIndex, label, split, split_group, cv_fold, orig_h, orig_w, n_die, n_fail, ..."""
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def _maps(path: str = str(LABELED)) -> pd.Series:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: run `python scripts/prepare_data.py` once to build it")
    return pd.read_pickle(p).set_index("uid")["waferMap"]


def get_wafer_map(uid: int) -> np.ndarray:
    """Raw wafer map (uint8, 0 = no die, 1 = pass, 2 = fail) for a uid."""
    return _maps().at[int(uid)]


def wafers_for_stream(split: str = "val") -> pd.DataFrame:
    """Wafers the factory simulation may stream, ordered by lot then wafer index.

    Spec 9.3: use "val" while building the demo; switch to "test" only after the single final
    test evaluation has been run, so nobody looks at test predictions beforehand.
    """
    if split not in ("val", "test"):
        raise ValueError("stream only val or test wafers: the model was trained on train")
    s = load_split()
    return s[s["split"] == split].sort_values(["lotName", "waferIndex"]).reset_index(drop=True)

"""Clean, encode and deduplicate wafer maps (spec Sections 3.4-3.6).

`encode_area` / `encode_nearest` are also what `wafer.infer` must call at serving time
(spec 9.1: preprocessing is never re-implemented), so keep them dependency-light and pure.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd

from wafer.constants import CLASS_TO_IDX, CLASSES, FAIL, SPLITS
from wafer.data.load import _flat

log = logging.getLogger(__name__)

DEFAULT_SIZE = 64


# --------------------------------------------------------------------------- 3.4 clean
@dataclass
class CleanInfo:
    n_raw: int = 0
    n_labeled: int = 0
    n_unlabeled: int = 0
    n_dropped_invalid: int = 0          # not 2D, or values outside {0,1,2}
    n_dropped_empty: int = 0            # no dies at all
    dropped_invalid_by_label: dict = field(default_factory=dict)
    n_lot_filled: int = 0
    n_wafer_index_missing: int = 0


def map_stats(m) -> tuple[bool, int, int, int, int]:
    """(valid, h, w, n_die, n_fail) for one raw map. Valid = 2D with values in {0,1,2}."""
    a = np.asarray(m)
    if a.ndim != 2 or a.size == 0:
        return False, 0, 0, 0, 0
    valid = bool(np.all((a == 0) | (a == 1) | (a == 2)))
    return valid, a.shape[0], a.shape[1], int(np.count_nonzero(a)), int(np.count_nonzero(a == FAIL))


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, CleanInfo]:
    """Return (labelled, unlabelled_meta, info).

    `uid` is the positional row number in the raw dataframe, so it is stable across reloads
    of the same file regardless of the pickle's index.
    """
    info = CleanInfo(n_raw=len(raw))
    df = pd.DataFrame({
        "uid": np.arange(len(raw), dtype=np.int64),
        "waferMap": raw["waferMap"].to_numpy(),
        "dieSize": pd.to_numeric(raw["dieSize"], errors="coerce").astype(float).to_numpy(),
        "lotName": raw["lotName"].to_numpy(dtype=object),
        "waferIndex": pd.to_numeric(raw["waferIndex"], errors="coerce").to_numpy(),
        "label": raw["failureType"].map(_flat).to_numpy(dtype=object),
        "orig_split": raw["trianTestLabel"].map(_flat).to_numpy(dtype=object),
    })

    stats = np.array([map_stats(m) for m in df["waferMap"]], dtype=np.int64).reshape(-1, 5)
    df["valid"] = stats[:, 0].astype(bool)
    df["orig_h"], df["orig_w"] = stats[:, 1], stats[:, 2]
    df["n_die"], df["n_fail"] = stats[:, 3], stats[:, 4]

    # Missing lot -> singleton lot, so it can never straddle splits.
    lot = df["lotName"]
    missing_lot = lot.isna() | (lot.astype(str).str.strip() == "")
    info.n_lot_filled = int(missing_lot.sum())
    df.loc[missing_lot, "lotName"] = "nolot_" + df.loc[missing_lot, "uid"].astype(str)
    df["lotName"] = df["lotName"].astype(str)

    wi = df["waferIndex"]
    info.n_wafer_index_missing = int(wi.isna().sum())
    df["waferIndex"] = wi.fillna(-1).astype(np.int64)

    labeled_mask = df["label"].notna()
    unl = df.loc[~labeled_mask]
    info.n_unlabeled = len(unl)
    unlabeled_meta = unl[["uid", "lotName", "waferIndex", "orig_h", "orig_w", "dieSize",
                          "n_die", "n_fail", "valid"]].reset_index(drop=True)

    lab = df.loc[labeled_mask].copy()
    unknown = set(lab["label"]) - set(CLASSES)
    if unknown:
        raise ValueError(f"Unknown labels in failureType: {sorted(unknown)}")

    invalid = ~lab["valid"]
    info.n_dropped_invalid = int(invalid.sum())
    info.dropped_invalid_by_label = lab.loc[invalid, "label"].value_counts().to_dict()
    empty = lab["valid"] & (lab["n_die"] == 0)
    info.n_dropped_empty = int(empty.sum())
    lab = lab.loc[~invalid & ~empty]

    missing_classes = set(CLASSES) - set(lab["label"])
    if missing_classes:
        raise ValueError(f"Classes with no labelled wafers: {sorted(missing_classes)}")

    lab["waferMap"] = [np.ascontiguousarray(m, dtype=np.uint8) for m in lab["waferMap"]]
    lab["label"] = lab["label"].astype(str)
    lab["label_idx"] = lab["label"].map(CLASS_TO_IDX).astype(np.int64)
    lab["hash"] = [hash_map(m) for m in lab["waferMap"]]
    lab = lab.drop(columns="valid").reset_index(drop=True)
    info.n_labeled = len(lab)
    return lab, unlabeled_meta, info


# --------------------------------------------------------------------------- 3.5 encode
def _check_map(m) -> np.ndarray:
    a = np.asarray(m)
    if a.ndim != 2 or a.size == 0:
        raise ValueError(f"wafer map must be a non-empty 2D array, got shape {a.shape}")
    return a


def validate_map(m) -> np.ndarray:
    """Strict check used at serving time (spec 9.1): 2D, integer values in {0,1,2}, >= 1 die.
    Returns the map as contiguous uint8; raises ValueError with a clear message otherwise."""
    a = np.asarray(m)
    if a.dtype == object or not (np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.floating)
                                 or a.dtype == bool):
        raise ValueError(f"wafer map must be numeric, got dtype {a.dtype}")
    a = _check_map(a)
    if not np.all((a == 0) | (a == 1) | (a == 2)):
        bad = np.unique(a[(a != 0) & (a != 1) & (a != 2)])[:5]
        raise ValueError(f"wafer map values must be 0 (no die), 1 (pass) or 2 (fail); found {bad.tolist()}")
    if not np.any(a > 0):
        raise ValueError("wafer map has no dies (all zeros)")
    return np.ascontiguousarray(a, dtype=np.uint8)


def encode_area(m, size: int = DEFAULT_SIZE) -> np.ndarray:
    """Encoding B: (2, S, S) float32 = [wafer_frac, fail_frac], each in [0, 1].

    INTER_AREA on float masks gives "fraction of dies in this block" when downsampling and
    behaves like nearest when upsampling, so one call handles every map size. Thin scratches
    survive because every die contributes to the output pixels it overlaps.
    """
    a = _check_map(m)
    wafer = (a > 0).astype(np.float32)
    fail = (a == FAIL).astype(np.float32)
    wafer_frac = cv2.resize(wafer, (size, size), interpolation=cv2.INTER_AREA)
    fail_frac = cv2.resize(fail, (size, size), interpolation=cv2.INTER_AREA)
    wafer_frac = np.clip(wafer_frac, 0.0, 1.0)
    fail_frac = np.minimum(np.clip(fail_frac, 0.0, 1.0), wafer_frac)  # keeps pass_frac >= 0
    return np.stack([wafer_frac, fail_frac])


def encode_nearest(m, size: int = DEFAULT_SIZE) -> np.ndarray:
    """Encoding A: (S, S) uint8 in {0,1,2}. Comparator only (E1): can drop thin scratches."""
    a = _check_map(m).astype(np.uint8, copy=False)
    return cv2.resize(a, (size, size), interpolation=cv2.INTER_NEAREST)


def area_to_uint8(x: np.ndarray) -> np.ndarray:
    """Store fractions as uint8 x255 (spec 3.8). Monotone, so fail <= wafer is preserved."""
    return np.rint(np.asarray(x) * 255.0).astype(np.uint8)


def area_channels(x_area_u8: np.ndarray) -> np.ndarray:
    """(..., 2, S, S) uint8 -> (..., 3, S, S) float32 [wafer_frac, pass_frac, fail_frac]."""
    x = np.asarray(x_area_u8, dtype=np.float32) / 255.0
    wafer, fail = x[..., 0, :, :], x[..., 1, :, :]
    return np.stack([wafer, np.clip(wafer - fail, 0.0, 1.0), fail], axis=-3)


def encode_all(maps, size: int = DEFAULT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Encode a sequence of raw maps -> (X_area uint8 (N,2,S,S), X_nearest uint8 (N,S,S))."""
    n = len(maps)
    x_area = np.empty((n, 2, size, size), dtype=np.uint8)
    x_near = np.empty((n, size, size), dtype=np.uint8)
    for i, m in enumerate(maps):
        x_area[i] = area_to_uint8(encode_area(m, size))
        x_near[i] = encode_nearest(m, size)
    return x_area, x_near


# --------------------------------------------------------------------------- 3.6 dedupe
def hash_map(m) -> str:
    a = np.ascontiguousarray(m, dtype=np.uint8)
    return hashlib.sha1(a.tobytes() + str(a.shape).encode()).hexdigest()


DUP_MIN_MINORITY = 1   # a map is "informative" if min(n_fail, n_pass) >= this


def informative(df: pd.DataFrame, min_minority: int = DUP_MIN_MINORITY) -> pd.Series:
    """Maps with both pass and fail dies. Identical *uniform* maps (all-pass, all-fail) are
    expected between different physical wafers of the same product, so they never link lots."""
    n_pass = df["n_die"] - df["n_fail"]
    return np.minimum(df["n_fail"], n_pass) >= min_minority


@dataclass
class DuplicateReport:
    n_groups: int
    n_rows_in_groups: int
    n_conflicting_groups: int
    n_rows_in_conflicting_groups: int
    n_informative_groups: int
    class_mix: dict


def assign_dup_groups(df: pd.DataFrame) -> pd.Series:
    """Integer id per duplicate group (hash seen more than once), -1 for unique maps."""
    counts = df["hash"].map(df["hash"].value_counts())
    dup_hashes = sorted(df.loc[counts > 1, "hash"].unique())
    ids = {h: i for i, h in enumerate(dup_hashes)}
    return df["hash"].map(ids).fillna(-1).astype(np.int64)


def duplicate_report(df: pd.DataFrame, min_minority: int = DUP_MIN_MINORITY) -> DuplicateReport:
    dups = df.loc[df["dup_group"] >= 0]
    g = dups.groupby("dup_group")
    conflicting = g["label"].nunique() > 1
    conflict_ids = conflicting[conflicting].index
    return DuplicateReport(
        n_groups=int(g.ngroups),
        n_rows_in_groups=len(dups),
        n_conflicting_groups=int(conflicting.sum()),
        n_rows_in_conflicting_groups=int(dups["dup_group"].isin(conflict_ids).sum()),
        n_informative_groups=int(informative(dups, min_minority).groupby(dups["dup_group"]).any().sum()),
        class_mix=dups["label"].value_counts().to_dict(),
    )


@dataclass
class LinkReport:
    n_cross_lot_hashes: int     # informative maps that appear in more than one lot
    n_lots_linked: int          # lots merged with at least one other lot
    n_groups: int               # split groups after linking
    largest_group_wafers: int
    largest_group_lots: int


def link_lots(df: pd.DataFrame, min_minority: int = DUP_MIN_MINORITY) -> tuple[pd.Series, LinkReport]:
    """Split group per row: lots that share an informative duplicate map are merged (union-find).

    Splitting by these groups puts every copy of a duplicated wafer in the same split, so no
    evaluation wafer has to be dropped and the split search sees the final class counts.
    The group id is the alphabetically first lotName in the group.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != root:          # path compression
            parent[x], x = root, parent[x]
        return root

    inf = df.loc[informative(df, min_minority), ["hash", "lotName"]]
    lots_per_hash = inf.groupby("hash")["lotName"].unique()
    cross = lots_per_hash[lots_per_hash.map(len) > 1]
    for lots in cross:
        lots = sorted(lots)
        for other in lots[1:]:
            a, b = find(lots[0]), find(other)
            if a != b:
                parent[max(a, b)] = min(a, b)

    group = df["lotName"].map(lambda lot: find(lot) if lot in parent else lot)
    sizes = df.groupby(group)["lotName"].agg(["size", "nunique"])
    linked = {lot for lots in cross for lot in lots}
    return group.rename("split_group"), LinkReport(
        n_cross_lot_hashes=len(cross), n_lots_linked=len(linked), n_groups=len(sizes),
        largest_group_wafers=int(sizes["size"].max()), largest_group_lots=int(sizes["nunique"].max()))


def cross_split_duplicates(df: pd.DataFrame, min_minority: int = DUP_MIN_MINORITY) -> pd.Series:
    """Rows whose informative map also appears in a different split. Must be empty after
    splitting by `link_lots` groups; checked as an assertion (spec 3.6)."""
    assert set(df["split"].unique()) <= set(SPLITS)
    inf = informative(df, min_minority)
    n_splits = df.loc[inf].groupby("hash")["split"].nunique()
    bad = set(n_splits[n_splits > 1].index)
    return inf & df["hash"].isin(bad)

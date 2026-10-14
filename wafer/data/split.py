"""Lot-grouped, stratified 70/15/15 split with an automatic, logged seed search (spec 3.7)."""
from __future__ import annotations

import logging
import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from wafer.constants import CLASSES, SPLITS

log = logging.getLogger(__name__)

N_FOLDS, N_TEST_FOLDS, N_VAL_FOLDS = 20, 3, 3      # 3/20 test, 3/20 val, 14/20 train
DEFAULT_MIN_COUNT = {c: 100 for c in CLASSES} | {"Near-full": 15, "Donut": 50}
DEFAULT_MAX_REL_DEV = 0.25
DEFAULT_PIN_TRAIN_FRAC = 0.01     # linked groups bigger than this share of wafers go to train
N_CV_FOLDS = 5


def make_split(y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    """One StratifiedGroupKFold(20) split -> array of 'train' / 'val' / 'test'."""
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold = np.empty(len(y), dtype=np.int64)
    with warnings.catch_warnings():
        # "least populated class has fewer members than n_splits": we check counts ourselves.
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        for k, (_, idx) in enumerate(sgkf.split(np.zeros(len(y)), y, groups=groups)):
            fold[idx] = k
    return np.select([fold < N_TEST_FOLDS, fold < N_TEST_FOLDS + N_VAL_FOLDS],
                     ["test", "val"], default="train").astype(object)


def split_counts(labels: np.ndarray, split: np.ndarray) -> pd.DataFrame:
    """Class x split wafer counts (all CLASSES and SPLITS present, zeros included)."""
    t = pd.crosstab(pd.Series(labels, name="label"), pd.Series(split, name="split"))
    return t.reindex(index=CLASSES, columns=SPLITS, fill_value=0)


def relative_deviation(counts: pd.DataFrame) -> pd.DataFrame:
    """|share in split - overall share| / overall share, per class and split."""
    overall = counts.sum(axis=1) / counts.to_numpy().sum()
    share = counts / counts.sum(axis=0)
    return share.sub(overall, axis=0).abs().div(overall, axis=0)


@dataclass
class SeedResult:
    seed: int
    feasible: bool
    worst_dev: float            # max relative deviation over classes x {val, test}
    worst_where: tuple[str, str]
    shortfalls: dict            # {(class, split): count below min} for infeasible seeds


def score_split(labels: np.ndarray, split: np.ndarray, min_count: dict) -> SeedResult:
    counts = split_counts(labels, split)
    dev = relative_deviation(counts)[["val", "test"]]
    where = dev.stack().idxmax()
    shortfalls = {(c, s): int(min_count.get(c, 0) - counts.at[c, s])
                  for c in CLASSES for s in ("val", "test")
                  if counts.at[c, s] < min_count.get(c, 0)}
    return SeedResult(seed=-1, feasible=not shortfalls, worst_dev=float(dev.to_numpy().max()),
                      worst_where=(str(where[0]), str(where[1])), shortfalls=shortfalls)


def make_split_pinned(y: np.ndarray, groups: np.ndarray, pin_train: np.ndarray, seed: int) -> np.ndarray:
    """make_split on the unpinned rows; pinned rows (oversized groups) always go to train."""
    split = np.full(len(y), "train", dtype=object)
    free = np.flatnonzero(~pin_train)
    split[free] = make_split(y[free], groups[free], seed)
    return split


def oversized_groups(groups, max_frac: float) -> np.ndarray:
    """Row mask for groups holding more than `max_frac` of all rows."""
    g = pd.Series(np.asarray(groups, dtype=object))
    sizes = g.map(g.value_counts())
    return (sizes > max_frac * len(g)).to_numpy()


def _try_seed(y: np.ndarray, groups: np.ndarray, labels: np.ndarray, min_count: dict,
              pin_train: np.ndarray, seed: int) -> SeedResult:
    r = score_split(labels, make_split_pinned(y, groups, pin_train, seed), min_count)
    r.seed = seed
    return r


@dataclass
class SplitSearch:
    seed: int
    split: np.ndarray
    chosen: SeedResult
    results: list[SeedResult]

    @property
    def n_feasible(self) -> int:
        return sum(r.feasible for r in self.results)


def search_split(labels, groups, seeds=range(100), min_count: dict | None = None,
                 jobs: int | None = None, pin_train=None) -> SplitSearch:
    """Try every seed; keep the feasible one with the lowest worst-case relative deviation.

    Feasible = every class meets `min_count` in both val and test. Ties go to the lower seed.
    `pin_train` (row mask) forces rows into train, e.g. oversized linked groups that would
    otherwise dominate a 15% evaluation split.
    """
    min_count = DEFAULT_MIN_COUNT if min_count is None else min_count
    labels = np.asarray(labels, dtype=object)
    y = pd.Categorical(labels, categories=CLASSES).codes.astype(np.int64)
    g = pd.factorize(np.asarray(groups, dtype=object))[0]
    pin = np.zeros(len(y), bool) if pin_train is None else np.asarray(pin_train, bool)
    seeds = list(seeds)
    jobs = jobs or os.cpu_count() or 1
    fn = partial(_try_seed, y, g, labels, min_count, pin)
    if jobs == 1 or len(seeds) == 1:
        results = [fn(s) for s in seeds]
    else:
        # spawn, not fork: forking a process that already runs BLAS/OpenMP threads can deadlock
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(jobs, len(seeds)), mp_context=ctx) as ex:
            results = list(ex.map(fn, seeds))

    feasible = [r for r in results if r.feasible]
    if not feasible:
        best = min(results, key=lambda r: (sum(r.shortfalls.values()), r.worst_dev))
        raise RuntimeError(
            f"No split seed out of {len(seeds)} meets the minimum per-class counts. "
            f"Closest: seed {best.seed}, short by {best.shortfalls}. Check lots per class in "
            "the data report, then lower the minimum with --min-count CLASS=N and log it."
        )
    chosen = min(feasible, key=lambda r: (r.worst_dev, r.seed))
    log.info("split seed %d: worst relative deviation %.3f (%s in %s); %d/%d seeds feasible",
             chosen.seed, chosen.worst_dev, *chosen.worst_where, len(feasible), len(seeds))
    return SplitSearch(seed=chosen.seed, split=make_split_pinned(y, g, pin, chosen.seed),
                       chosen=chosen, results=results)


def assign_cv_folds(labels, groups, split, seed: int = 0, n_folds: int = N_CV_FOLDS) -> np.ndarray:
    """cv_fold 0..n_folds-1 over train+val groups (grouped, stratified); -1 on test."""
    labels, groups, split = (np.asarray(a, dtype=object) for a in (labels, groups, split))
    out = np.full(len(labels), -1, dtype=np.int64)
    tv = np.flatnonzero(split != "test")
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        for k, (_, idx) in enumerate(sgkf.split(np.zeros(len(tv)), labels[tv], groups=groups[tv])):
            out[tv[idx]] = k
    return out


def check_split(df: pd.DataFrame, min_count: dict | None = None,
                max_rel_dev: float = DEFAULT_MAX_REL_DEV) -> pd.DataFrame:
    """Spec 3.7 assertions. Returns the relative-deviation table; raises AssertionError."""
    min_count = DEFAULT_MIN_COUNT if min_count is None else min_count
    assert set(df["split"].unique()) <= set(SPLITS), f"bad split values {df['split'].unique()}"

    for key in ("lotName", "split_group"):
        if key in df:
            per = df.groupby(key)["split"].nunique()
            leaked = per[per > 1]
            assert leaked.empty, f"{len(leaked)} {key} values appear in more than one split, e.g. {list(leaked.index[:5])}"
    if {"hash", "n_die", "n_fail"} <= set(df.columns):
        from wafer.data.preprocess import cross_split_duplicates
        n_dup = int(cross_split_duplicates(df).sum())
        assert n_dup == 0, f"{n_dup} wafers share an informative map with a wafer in another split"

    counts = split_counts(df["label"].to_numpy(), df["split"].to_numpy())
    short = {(c, s): int(counts.at[c, s]) for c in CLASSES for s in ("val", "test")
             if counts.at[c, s] < min_count.get(c, 0)}
    assert not short, f"classes below MIN_COUNT (class, split) -> count: {short}"

    dev = relative_deviation(counts)
    worst = dev.stack().idxmax()
    assert dev.to_numpy().max() <= max_rel_dev, (
        f"relative class-share deviation {dev.to_numpy().max():.3f} > {max_rel_dev} "
        f"({worst[0]} in {worst[1]}); see lots per class, raise --max-rel-dev only with a log entry"
    )

    if "cv_fold" in df:
        test = df["split"] == "test"
        assert (df.loc[test, "cv_fold"] == -1).all(), "test rows must have cv_fold == -1"
        assert df.loc[~test, "cv_fold"].between(0, N_CV_FOLDS - 1).all(), "cv_fold out of range"
        key = "split_group" if "split_group" in df else "lotName"
        cv_leak = df.loc[~test].groupby(key)["cv_fold"].nunique()
        assert (cv_leak <= 1).all(), f"a {key} appears in more than one cv_fold"
    return dev

"""`docs/results/data_report.md` writer (spec 3.8) and reader for the recorded SHA-256."""
from __future__ import annotations

import datetime as dt
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd

from wafer.constants import CLASSES, SPLITS

_SHA_RE = re.compile(r"SHA-256 of `LSWMD\.pkl`:\*\*\s*`([0-9a-fA-F]{64})`")


def recorded_sha256(report_path: str | Path) -> str | None:
    """The SHA-256 recorded by whoever first ran the pipeline, or None if not recorded yet."""
    p = Path(report_path)
    if not p.exists():
        return None
    m = _SHA_RE.search(p.read_text())
    return m.group(1).lower() if m else None


def _versions() -> str:
    import cv2
    import sklearn
    return (f"python {platform.python_version()} · numpy {np.__version__} · pandas {pd.__version__} "
            f"· opencv {cv2.__version__} · scikit-learn {sklearn.__version__}")


def _git_commit() -> str:
    from wafer.env import git_commit
    commit, dirty = git_commit()
    return commit + (" (dirty)" if dirty else "")


def _md_table(df: pd.DataFrame, index: bool = True) -> str:
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in d.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:.3f}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return str(v)


def split_table(meta: pd.DataFrame) -> pd.DataFrame:
    wafers = pd.crosstab(meta["label"], meta["split"]).reindex(index=CLASSES, columns=SPLITS, fill_value=0)
    lots = (meta.groupby(["label", "split"])["lotName"].nunique().unstack(fill_value=0)
            .reindex(index=CLASSES, columns=SPLITS, fill_value=0))
    t = pd.DataFrame(index=pd.Index(CLASSES, name="class"))
    for s in SPLITS:
        t[f"{s} wafers"] = wafers[s]
    t["total wafers"] = wafers.sum(axis=1)
    for s in SPLITS:
        t[f"{s} lots"] = lots[s]
    t["total lots"] = meta.groupby("label")["lotName"].nunique().reindex(CLASSES, fill_value=0)
    t["share %"] = 100.0 * t["total wafers"] / t["total wafers"].sum()
    totals = t.sum(numeric_only=True)
    totals[[f"{s} lots" for s in SPLITS] + ["total lots"]] = (
        [meta.loc[meta["split"] == s, "lotName"].nunique() for s in SPLITS] + [meta["lotName"].nunique()])
    t.loc["**all**"] = totals
    int_cols = [c for c in t.columns if c != "share %"]
    t[int_cols] = t[int_cols].round().astype(np.int64)
    return t


def write_report(path: str | Path, *, sha256: str, sha_verified: bool, clean_info, dup_report,
                 link_report, dup_min_minority: int, search, min_count: dict,
                 n_pinned: int = 0, pin_train_frac: float = 0.01,
                 max_rel_dev: float, rel_dev: pd.DataFrame, cv_seed: int, meta: pd.DataFrame,
                 sizes: list[int], command: str) -> None:
    ci = clean_info
    dr = dup_report
    lr = link_report
    ch = search.chosen
    area = (meta["orig_h"] * meta["orig_w"]).astype(np.int64)
    q = np.quantile(area, [0, 0.25, 0.5, 0.75, 1.0])
    shapes = (meta.groupby(["orig_h", "orig_w"]).size().sort_values(ascending=False).head(10)
              .rename("wafers").reset_index())
    bins = pd.qcut(area, 4, duplicates="drop")
    size_bins = bins.value_counts(sort=False).rename("wafers").rename_axis("orig_h*orig_w bin").reset_index()
    size_bins["orig_h*orig_w bin"] = size_bins["orig_h*orig_w bin"].astype(str)
    cv = (pd.crosstab(meta.loc[meta.cv_fold >= 0, "label"], meta.loc[meta.cv_fold >= 0, "cv_fold"])
          .reindex(index=CLASSES, fill_value=0))
    cv.columns = [f"fold {c}" for c in cv.columns]
    orig = pd.crosstab(meta["orig_split"].fillna("missing"), meta["split"]
                       ).reindex(columns=SPLITS, fill_value=0)
    dev_tbl = rel_dev.rename(columns=lambda s: f"{s} rel. dev.")

    text = f"""# Data report — WM-811K (Phase 1)

Generated {dt.datetime.now().astimezone().strftime('%a %d %b %Y %H:%M %Z')} · git `{_git_commit()}`
Command: `{command}`
Versions: {_versions()}

## File
- **SHA-256 of `LSWMD.pkl`:** `{sha256}`
- {'Verified against the hash recorded in this report before unpickling.' if sha_verified else '**First recording** — no hash was on record, so the file was unpickled unverified. Commit this report so everyone else is checked against it.'}

## Counts
| | wafers |
|---|---|
| raw rows | {ci.n_raw:,} |
| labelled | {ci.n_labeled + ci.n_dropped_invalid + ci.n_dropped_empty:,} |
| unlabelled (metadata kept for E9) | {ci.n_unlabeled:,} |
| dropped: not 2D / values outside {{0,1,2}} | {ci.n_dropped_invalid:,} {ci.dropped_invalid_by_label or ''} |
| dropped: no dies at all | {ci.n_dropped_empty:,} |
| **final labelled wafers (npz + metadata)** | **{len(meta):,}** |
| missing `lotName` filled as singleton lots | {ci.n_lot_filled:,} |
| missing `waferIndex` (set to -1) | {ci.n_wafer_index_missing:,} |

Encodings written for sizes: {', '.join(f'`wafers_s{s}.npz`' for s in sizes)}.

## Split (lot-grouped, stratified, 70/15/15)
- Actual shares: train {(meta.split == 'train').mean():.1%} · val {(meta.split == 'val').mean():.1%} · test {(meta.split == 'test').mean():.1%}.
- Seed search over {len(search.results)} seeds: **{search.n_feasible}** met the minimum counts. Chosen seed **{search.seed}**, worst relative class-share deviation in val/test **{ch.worst_dev:.3f}** ({ch.worst_where[0]} in {ch.worst_where[1]}).
- Minimum per-class counts in val and test: {min_count}. Maximum allowed relative deviation: {max_rel_dev}.
- Split unit: **split group** = a lot, or several lots linked because they share a duplicate map (see Duplicates). {lr.n_groups:,} groups.
- Lots per class decide how wide the per-class confidence intervals are. Bootstrap by `split_group`.

{_md_table(split_table(meta))}

Relative class-share deviation per split (|share in split − overall share| / overall share):

{_md_table(dev_tbl.rename_axis('class'))}

### CV folds (train + val lots, seed {cv_seed})
{_md_table(cv.rename_axis('class'))}

## Duplicates (exact hash of the original map)
- Duplicate groups: **{dr.n_groups:,}** covering {dr.n_rows_in_groups:,} wafers; {dr.n_informative_groups:,} groups are *informative* (min(fail dies, pass dies) ≥ {dup_min_minority}). Uniform maps (all-pass, all-fail) are expected to repeat between different wafers and are ignored.
- **Groups with conflicting labels: {dr.n_conflicting_groups:,}** ({dr.n_rows_in_conflicting_groups:,} wafers). Identical maps with different labels are a measured lower bound on label noise.
- Class mix of duplicated wafers: {dr.class_mix}
- **Leakage handling:** {lr.n_cross_lot_hashes:,} informative maps appear in more than one lot. Their {lr.n_lots_linked:,} lots were linked into shared split groups, so every copy lands in the same split and no wafer was dropped. Largest group: {lr.largest_group_wafers:,} wafers / {lr.largest_group_lots:,} lots. Verified after splitting: no informative map appears in two splits.
- **Pinned to train:** {n_pinned:,} wafers in linked groups larger than {100 * pin_train_frac:.0f}% of the data. A group that big would otherwise dominate a 15% val or test split and skew its product mix.

## Map sizes (final labelled set)
- Distinct shapes: {meta.groupby(['orig_h', 'orig_w']).ngroups:,}. `orig_h*orig_w`: min {q[0]:,.0f} · q25 {q[1]:,.0f} · median {q[2]:,.0f} · q75 {q[3]:,.0f} · max {q[4]:,.0f}.

{_md_table(size_bins, index=False)}

Most common shapes:

{_md_table(shapes, index=False)}

## Original `trianTestLabel` tag vs our split (reference only — never used for evaluation)
{_md_table(orig.rename_axis('orig_split'))}
"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)

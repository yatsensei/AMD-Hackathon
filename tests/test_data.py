"""Phase 1 tests (spec Section 11). Synthetic-data tests always run; real-data tests run only
once `scripts/prepare_data.py` has produced data/processed/ and docs/results/data_report.md."""
from __future__ import annotations

import json
import pickle
import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.synthetic import circle_wafer, make_raw_df
from wafer.constants import CLASS_TO_IDX, CLASSES, SPLITS
from wafer.data import preprocess, split as splitlib
from wafer.data.load import _flat, load_raw, read_pickle_with_aliases, sha256_file
from wafer.data.pipeline import META_COLUMNS, main as prepare_main
from wafer.data.report import recorded_sha256

ROOT = Path(__file__).resolve().parents[1]
SMALL_MIN = {c: 2 for c in CLASSES}


# --------------------------------------------------------------------------- load
@pytest.mark.parametrize("x, expected", [
    ([["x"]], "x"), (["x"], "x"), ([], None), (np.array([["Edge-Ring"]]), "Edge-Ring"),
    (np.zeros((0, 0)), None), (np.array([], dtype=object), None), (None, None),
    (float("nan"), None), ("none", "none"),
])
def test_flat(x, expected):
    assert _flat(x) == expected


def test_sha_mismatch_refuses_before_unpickling(tmp_path):
    p = tmp_path / "LSWMD.pkl"
    p.write_bytes(b"not a pickle at all")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_raw(p, expected_sha256="0" * 64)


def test_old_module_paths_are_aliased(tmp_path, monkeypatch):
    """Simulate a pickle that references a module path that no longer exists."""
    new = types.ModuleType("wafer_test_newpkg")
    exec("class Thing:\n    def __init__(self, v): self.v = v", new.__dict__)
    new.Thing.__module__ = "wafer_test_oldpkg"
    monkeypatch.setitem(sys.modules, "wafer_test_oldpkg", new)
    payload = pickle.dumps(new.Thing(7))
    del sys.modules["wafer_test_oldpkg"]
    monkeypatch.setitem(sys.modules, "wafer_test_newpkg", new)
    p = tmp_path / "old.pkl"
    p.write_bytes(payload)

    with pytest.raises(ModuleNotFoundError):
        read_pickle_with_aliases(p, prefixes={})
    obj = read_pickle_with_aliases(p, prefixes={"wafer_test_oldpkg": "wafer_test_newpkg"})
    assert obj.v == 7 and sys.modules["wafer_test_oldpkg"] is new
    monkeypatch.delitem(sys.modules, "wafer_test_oldpkg")


def test_python2_pickle_loads_with_latin1(tmp_path):
    """LSWMD.pkl is a Python 2 pickle: a py2 `str` with non-ASCII bytes must still load."""
    p = tmp_path / "py2.pkl"
    p.write_bytes(b"\x80\x02U\x02\x9a\x9bq\x00.")       # PROTO 2, SHORT_BINSTRING b'\x9a\x9b', STOP
    assert read_pickle_with_aliases(p) == "\x9a\x9b"


def test_load_raw_roundtrip_with_hash(tmp_path):
    df = make_raw_df(n_lots=10, wafers_per_lot=3)
    p = tmp_path / "LSWMD.pkl"
    df.to_pickle(p)
    out = load_raw(p, expected_sha256=sha256_file(p))
    assert list(out.columns) == ["waferMap", "dieSize", "lotName", "waferIndex", "trianTestLabel", "failureType"]


# --------------------------------------------------------------------------- clean
@pytest.fixture(scope="module")
def raw():
    return make_raw_df()


@pytest.fixture(scope="module")
def cleaned(raw):
    return preprocess.clean(raw)


def test_clean_counts_and_columns(raw, cleaned):
    lab, unl, info = cleaned
    n_labelled_raw = sum(_flat(x) is not None for x in raw["failureType"])
    assert info.n_raw == len(raw)
    assert info.n_labeled + info.n_dropped_invalid + info.n_dropped_empty == n_labelled_raw
    assert info.n_unlabeled == len(unl) == len(raw) - n_labelled_raw
    assert info.n_dropped_invalid == 1 and info.n_dropped_empty == 1 and info.n_lot_filled == 1
    assert set(lab["label"]) == set(CLASSES)
    for col in ["uid", "lotName", "waferIndex", "dieSize", "label", "label_idx", "orig_split",
                "orig_h", "orig_w", "n_die", "n_fail", "hash", "waferMap"]:
        assert col in lab
    assert lab["uid"].is_unique and lab["uid"].is_monotonic_increasing
    assert (lab["label_idx"] == lab["label"].map(CLASS_TO_IDX)).all()
    assert lab["waferIndex"].dtype == np.int64 and lab["dieSize"].dtype == float


def test_clean_uid_is_raw_row_and_shapes_match(raw, cleaned):
    lab, _, _ = cleaned
    for row in lab.sample(20, random_state=0).itertuples():
        m = raw.at[row.uid, "waferMap"]
        assert np.array_equal(row.waferMap, m)
        assert (row.orig_h, row.orig_w) == m.shape
        assert row.n_fail == (m == 2).sum() and row.n_die == (m > 0).sum()


def test_clean_fills_missing_lot_as_singleton(cleaned):
    lab, _, _ = cleaned
    filled = lab[lab["lotName"].str.startswith("nolot_")]
    assert len(filled) == 1
    assert filled["lotName"].iloc[0] == f"nolot_{filled['uid'].iloc[0]}"


def test_clean_rejects_unknown_label(raw):
    bad = raw.copy()
    bad.at[0, "failureType"] = np.array([["Unknown-Pattern"]], dtype=object)
    with pytest.raises(ValueError, match="Unknown labels"):
        preprocess.clean(bad)


# --------------------------------------------------------------------------- encode
@pytest.mark.parametrize("shape", [(10, 10), (64, 64), (300, 200), (3, 5), (25, 90)])
@pytest.mark.parametrize("size", [64, 96])
def test_encodings_shape_and_values(shape, size):
    rng = np.random.default_rng(0)
    m = rng.integers(0, 3, size=shape).astype(np.uint8)
    a = preprocess.encode_area(m, size)
    assert a.shape == (2, size, size) and a.dtype == np.float32
    assert a.min() >= 0 and a.max() <= 1 and (a[1] <= a[0]).all()
    n = preprocess.encode_nearest(m, size)
    assert n.shape == (size, size) and set(np.unique(n)) <= {0, 1, 2}
    u8 = preprocess.area_to_uint8(a)
    ch = preprocess.area_channels(u8)
    assert ch.shape == (3, size, size) and ch.min() >= 0 and ch.max() <= 1
    assert np.allclose(ch[0], ch[1] + ch[2], atol=1e-6)            # wafer = pass + fail


def test_encode_rejects_bad_maps():
    for bad in [np.zeros((0, 5)), np.zeros(5), np.zeros((2, 2, 2))]:
        with pytest.raises(ValueError):
            preprocess.encode_area(bad)


def _scratch_survival(offset: int, size: int = 64):
    H = W = 200
    m, _ = circle_wafer(H, W)
    idx = np.arange(30, 170)
    m[idx, idx + offset] = 2
    fail_u8 = preprocess.area_to_uint8(preprocess.encode_area(m, size))[1]
    near = preprocess.encode_nearest(m, size)
    # the output pixel containing each scratch die's centre
    py = ((idx + 0.5) * size / H).astype(int)
    px = ((idx + offset + 0.5) * size / W).astype(int)
    area_kept = np.mean(fail_u8[py, px] > 0)
    near_kept = (near == 2).sum() / len(idx)
    return area_kept, near_kept


@pytest.mark.parametrize("offset", [0, 1])
def test_scratch_survives_area_encoding(offset):
    """Why encoding B exists (spec 3.5): a 1-die-wide scratch must survive downsampling,
    even after uint8 storage. Nearest-neighbour keeps a fraction at best, none when shifted."""
    area_kept, near_kept = _scratch_survival(offset)
    assert area_kept == 1.0
    assert near_kept < 0.5


def test_nearest_loses_offset_scratch():
    _, near_kept = _scratch_survival(offset=1)
    assert near_kept < 0.1


# --------------------------------------------------------------------------- dedupe
def test_hash_depends_on_shape():
    a, b = np.zeros((2, 3), np.uint8), np.zeros((3, 2), np.uint8)
    assert preprocess.hash_map(a) != preprocess.hash_map(b)
    assert preprocess.hash_map(a) == preprocess.hash_map(a.copy())
    assert preprocess.hash_map(a.astype(np.int64)) == preprocess.hash_map(a)


def test_informative_excludes_uniform_maps():
    df = pd.DataFrame({"n_die": [100, 100, 100, 100], "n_fail": [0, 100, 1, 99]})
    assert preprocess.informative(df).tolist() == [False, False, True, True]
    assert preprocess.informative(df, min_minority=2).tolist() == [False, False, False, False]


def test_link_lots_merges_lots_sharing_informative_maps():
    df = pd.DataFrame({
        "lotName": ["A", "B", "B", "C", "D", "E", "F"],
        "hash":    ["h1", "h1", "h2", "h2", "u", "u", "x"],
        "n_die":   [10] * 7,
        "n_fail":  [3, 3, 4, 4, 0, 0, 2],      # D/E share an all-pass map: must NOT link
    })
    group, rep = preprocess.link_lots(df)
    assert group.tolist() == ["A", "A", "A", "A", "D", "E", "F"]       # A-B-C chained
    assert (rep.n_cross_lot_hashes, rep.n_lots_linked, rep.n_groups) == (2, 3, 4)
    assert (rep.largest_group_wafers, rep.largest_group_lots) == (4, 3)


def test_cross_split_duplicates_detects_only_informative_leaks():
    df = pd.DataFrame({"hash": ["f", "f", "z", "z", "g", "g"],
                       "split": ["train", "test", "train", "val", "val", "val"],
                       "n_die": [10] * 6, "n_fail": [3, 3, 0, 0, 5, 5]})
    assert preprocess.cross_split_duplicates(df).tolist() == [True, True, False, False, False, False]


def test_duplicate_report_counts_conflicts():
    df = pd.DataFrame({"hash": ["a", "a", "b", "b", "c"], "label": ["Loc", "Loc", "Loc", "Center", "none"],
                       "n_die": [5] * 5, "n_fail": [1, 1, 0, 0, 0]})
    df["dup_group"] = preprocess.assign_dup_groups(df)
    assert df["dup_group"].tolist()[-1] == -1
    r = preprocess.duplicate_report(df)
    assert (r.n_groups, r.n_rows_in_groups, r.n_conflicting_groups, r.n_informative_groups) == (2, 4, 1, 1)


# --------------------------------------------------------------------------- split
@pytest.fixture(scope="module")
def searched(cleaned):
    lab, _, _ = cleaned
    lab = lab.assign(split_group=preprocess.link_lots(lab)[0])
    s = splitlib.search_split(lab["label"].to_numpy(), lab["split_group"].to_numpy(), seeds=range(6),
                              min_count=SMALL_MIN, jobs=1)
    return lab.assign(split=s.split), s


def test_split_no_lot_overlap_and_min_counts(searched):
    df, s = searched
    assert df.groupby("lotName")["split"].nunique().max() == 1
    counts = splitlib.split_counts(df["label"].to_numpy(), df["split"].to_numpy())
    assert (counts[["val", "test"]] >= 2).all().all()
    frac = df["split"].value_counts(normalize=True)
    assert abs(frac["test"] - 0.15) < 0.05 and abs(frac["val"] - 0.15) < 0.05


def test_split_search_picks_lowest_feasible_and_is_deterministic(cleaned, searched):
    lab, _, _ = cleaned
    _, s = searched
    feasible = [r for r in s.results if r.feasible]
    assert s.chosen.worst_dev == min(r.worst_dev for r in feasible)
    groups = preprocess.link_lots(lab)[0]
    again = splitlib.make_split(pd.Categorical(lab["label"], categories=CLASSES).codes,
                                pd.factorize(groups)[0], s.seed)
    assert (again == s.split).all()


def test_split_search_parallel_matches_serial(cleaned, searched):
    lab, _, _ = cleaned
    _, s = searched
    p = splitlib.search_split(lab["label"].to_numpy(), preprocess.link_lots(lab)[0].to_numpy(), seeds=range(6),
                              min_count=SMALL_MIN, jobs=2)
    assert p.seed == s.seed and (p.split == s.split).all()


def test_oversized_groups_are_pinned_to_train(cleaned):
    lab, _, _ = cleaned
    groups = lab["lotName"].to_numpy().copy()
    groups[:300] = "giant"                                     # one group with ~15% of rows
    pin = splitlib.oversized_groups(groups, 0.05)
    assert pin[:300].all() and not pin[300:].any()
    s = splitlib.search_split(lab["label"].to_numpy(), groups, seeds=range(3), min_count={c: 1 for c in CLASSES},
                              jobs=1, pin_train=pin)
    assert (s.split[:300] == "train").all()


def test_split_search_infeasible_raises(cleaned):
    lab, _, _ = cleaned
    with pytest.raises(RuntimeError, match="minimum per-class counts"):
        splitlib.search_split(lab["label"].to_numpy(), lab["lotName"].to_numpy(), seeds=range(2),
                              min_count={c: 10**6 for c in CLASSES}, jobs=1)


def test_cv_folds_cover_train_val_only(searched):
    df, _ = searched
    df = df.assign(cv_fold=splitlib.assign_cv_folds(df["label"], df["split_group"], df["split"], seed=0))
    test = df["split"] == "test"
    assert (df.loc[test, "cv_fold"] == -1).all()
    assert set(df.loc[~test, "cv_fold"]) == set(range(splitlib.N_CV_FOLDS))
    assert df.loc[~test].groupby("split_group")["cv_fold"].nunique().max() == 1
    splitlib.check_split(df, min_count=SMALL_MIN, max_rel_dev=10.0)


def test_check_split_catches_lot_leak(searched):
    df, _ = searched
    sizes = df["lotName"].value_counts()
    lot = sizes.index[sizes >= 2][0]
    j = df.index[df["lotName"] == lot][0]
    bad = df.copy()
    bad.at[j, "split"] = next(s for s in SPLITS if s != bad.at[j, "split"])
    with pytest.raises(AssertionError, match="more than one split"):
        splitlib.check_split(bad, min_count={}, max_rel_dev=10.0)


def test_relative_deviation_zero_for_identical_shares():
    labels = np.array(CLASSES * 3, dtype=object)
    split = np.repeat(SPLITS, len(CLASSES)).astype(object)
    dev = splitlib.relative_deviation(splitlib.split_counts(labels, split))
    assert np.allclose(dev.to_numpy(), 0)


# --------------------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("phase1")
    raw = make_raw_df()
    raw.to_pickle(d / "LSWMD.pkl")
    args = ["--raw", str(d / "LSWMD.pkl"), "--interim-dir", str(d / "interim"),
            "--out-dir", str(d / "processed"), "--report", str(d / "data_report.md"),
            "--sizes", "64", "32", "--split-seeds", "4", "--jobs", "1", "--max-rel-dev", "10",
            "--yolo", "--yolo-dir", str(d / "yolo"), "--yolo-size", "32"]
    args += sum((["--min-count", f"{c}=2"] for c in CLASSES), [])
    meta = prepare_main(args)
    return d, raw, meta, args


def test_pipeline_outputs_align(pipeline_run):
    d, _, meta, _ = pipeline_run
    meta_disk = pd.read_parquet(d / "processed" / "metadata.parquet")
    assert list(meta_disk.columns) == META_COLUMNS
    for s in (64, 32):
        z = np.load(d / "processed" / f"wafers_s{s}.npz")
        assert z["X_area"].shape == (len(meta_disk), 2, s, s) and z["X_area"].dtype == np.uint8
        assert z["X_nearest"].shape == (len(meta_disk), s, s) and z["X_nearest"].dtype == np.uint8
        assert (z["uid"] == meta_disk["uid"].to_numpy()).all()
        assert (z["y"] == meta_disk["label_idx"].to_numpy()).all()
        assert (z["X_area"][:, 1] <= z["X_area"][:, 0]).all()
    assert (d / "interim" / "labeled.pkl").exists() and (d / "interim" / "unlabeled_meta.parquet").exists()


def test_pipeline_encoding_matches_raw(pipeline_run):
    """Train/serve parity at the data level: npz rows equal encoding the raw map directly."""
    d, raw, meta, _ = pipeline_run
    z = np.load(d / "processed" / "wafers_s64.npz")
    for i in np.random.default_rng(0).choice(len(meta), 10, replace=False):
        m = raw.at[int(z["uid"][i]), "waferMap"]
        assert (z["X_area"][i] == preprocess.area_to_uint8(preprocess.encode_area(m, 64))).all()
        assert (z["X_nearest"][i] == preprocess.encode_nearest(m, 64)).all()


def test_pipeline_duplicates_never_straddle_splits(pipeline_run):
    """The synthetic data plants copies of real maps in new lots: they must be linked into the
    original's split group and land in the same split, with nothing dropped."""
    d, raw, _, _ = pipeline_run
    meta = pd.read_parquet(d / "processed" / "metadata.parquet")
    assert not preprocess.cross_split_duplicates(meta).any()
    planted = meta[meta.lotName.str.startswith("duplot")]
    assert len(planted) > 0
    inf = planted[preprocess.informative(planted)]
    assert len(inf) > 0
    for row in inf.itertuples():
        twins = meta[(meta.hash == row.hash) & (meta.uid != row.uid)]
        assert len(twins) and (twins.split == row.split).all() and (twins.split_group == row.split_group).all()
    n_valid_labelled = len(preprocess.clean(raw)[0])
    assert len(meta) == n_valid_labelled                        # nothing dropped by dedupe


def test_pipeline_report_records_and_then_enforces_sha(pipeline_run):
    d, _, _, args = pipeline_run
    report = (d / "data_report.md").read_text()
    sha = sha256_file(d / "LSWMD.pkl")
    assert recorded_sha256(d / "data_report.md") == sha
    assert "First recording" in report
    # A different raw file must now be refused before unpickling.
    other = make_raw_df(n_lots=60, seed=1)
    other.to_pickle(d / "LSWMD.pkl")
    try:
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            prepare_main(args)
    finally:
        make_raw_df().to_pickle(d / "LSWMD.pkl")


def test_pipeline_yolo_pngs(pipeline_run):
    from PIL import Image
    d, _, meta, _ = pipeline_run
    pngs = list((d / "yolo").rglob("*.png"))
    assert len(pngs) == len(meta)
    row = meta.iloc[0]
    img = Image.open(d / "yolo" / row.split / row.label / f"{row.uid}.png")
    assert img.size == (32, 32) and img.mode == "RGB"


# --------------------------------------------------------------------------- real data
REAL_META = ROOT / "data" / "processed" / "metadata.parquet"
REAL_NPZ = ROOT / "data" / "processed" / "wafers_s64.npz"
REAL_REPORT = ROOT / "docs" / "results" / "data_report.md"
real = pytest.mark.skipif(not (REAL_META.exists() and REAL_NPZ.exists() and REAL_REPORT.exists()),
                          reason="run scripts/prepare_data.py on the real LSWMD.pkl first")


@real
def test_real_labelled_count_matches_report():
    meta = pd.read_parquet(REAL_META)
    m = re.search(r"final labelled wafers \(npz \+ metadata\)\*\* \| \*\*([\d,]+)\*\*", REAL_REPORT.read_text())
    assert m and int(m.group(1).replace(",", "")) == len(meta)


@real
def test_real_npz_aligns_with_metadata():
    meta = pd.read_parquet(REAL_META)
    z = np.load(REAL_NPZ)
    assert (z["uid"] == meta["uid"].to_numpy()).all() and (z["y"] == meta["label_idx"].to_numpy()).all()


@real
def test_real_split_assertions():
    meta = pd.read_parquet(REAL_META)
    params = json.loads((REAL_META.parent / "split_params.json").read_text())
    splitlib.check_split(meta, min_count=params["min_count"], max_rel_dev=params["max_rel_dev"])
    assert not preprocess.cross_split_duplicates(meta, params["dup_min_minority"]).any()


def test_committed_split_file_matches_metadata():
    from wafer.data.access import SPLIT_FILE, load_split, wafers_for_stream
    s = load_split()
    assert len(s) == 172_950 and set(s["split"]) == {"train", "val", "test"}
    assert s.groupby("lotName")["split"].nunique().max() == 1
    w = wafers_for_stream("val")
    assert (w["split"] == "val").all() and len(w) == (s["split"] == "val").sum()
    with pytest.raises(ValueError):
        wafers_for_stream("train")
    if REAL_META.exists():
        meta = pd.read_parquet(REAL_META)
        assert (meta["uid"].to_numpy() == s["uid"].to_numpy()).all()
        assert (meta["split"].to_numpy() == s["split"].to_numpy()).all()
    assert SPLIT_FILE.exists()

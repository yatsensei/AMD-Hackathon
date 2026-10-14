"""Load the raw WM-811K pickle (spec Section 3.3).

`LSWMD.pkl` was written by an old pandas and references module paths that no longer
exist (e.g. `pandas.indexes`). We alias those paths to their modern locations and retry.
Pickles can execute code when loaded, so the file's SHA-256 is checked *before* unpickling
whenever an expected hash is known (spec Rule 0.6).
"""
from __future__ import annotations

import hashlib
import importlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Old module path prefix -> modern prefix. Extend this if a new ModuleNotFoundError names
# another old path, and record the patch in the spec's Decisions Log.
OLD_MODULE_PREFIXES = {
    "pandas.indexes": "pandas.core.indexes",
}
_MAX_ALIAS_ROUNDS = 10


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _alias_module(old: str, new: str) -> bool:
    """Register `old` in sys.modules as the already-imported module `new`.

    Aliasing to the *same* module object matters: letting Python import `old` afresh would
    create duplicate classes, and isinstance checks inside pandas would then fail.
    """
    try:
        sys.modules[old] = importlib.import_module(new)
    except ImportError:
        return False
    log.warning("pickle compat: aliased module %s -> %s", old, new)
    return True


def _modern_name(missing: str, prefixes: dict[str, str]) -> str | None:
    for old, new in prefixes.items():
        if missing == old or missing.startswith(old + "."):
            return new + missing[len(old):]
    return None


def _read_pickle(path: str | Path):
    """`pd.read_pickle`, falling back to latin1 for Python 2 pickles.

    LSWMD.pkl was written by Python 2: its byte strings only decode as latin1, and pandas >= 3
    no longer retries that by itself. pandas' compat Unpickler also remaps old class locations.
    """
    try:
        return pd.read_pickle(path)
    except UnicodeDecodeError:
        log.warning("pickle compat: Python 2 pickle, loading with encoding='latin1'")
        try:
            from pandas.compat.pickle_compat import Unpickler
        except ImportError:                                  # pragma: no cover
            from pickle import Unpickler
        with open(path, "rb") as f:
            return Unpickler(f, encoding="latin1").load()


def read_pickle_with_aliases(path: str | Path, prefixes: dict[str, str] | None = None):
    """Read a pickle, aliasing old module paths one at a time until it loads."""
    prefixes = OLD_MODULE_PREFIXES if prefixes is None else prefixes
    for _ in range(_MAX_ALIAS_ROUNDS):
        try:
            return _read_pickle(path)
        except ModuleNotFoundError as e:
            missing = e.name or ""
            new = _modern_name(missing, prefixes)
            if new is None or not _alias_module(missing, new):
                raise ModuleNotFoundError(
                    f"{path}: old module {missing!r} has no known modern equivalent. Add a "
                    f"prefix mapping to OLD_MODULE_PREFIXES in {__name__} (or try pandas<2) "
                    "and log the patch in the spec's Decisions Log."
                ) from e
    raise RuntimeError(f"{path}: gave up after {_MAX_ALIAS_ROUNDS} module aliases")


def load_raw(path: str | Path = "data/raw/LSWMD.pkl",
             expected_sha256: str | None = None) -> pd.DataFrame:
    """Load LSWMD.pkl. If `expected_sha256` is given, verify it before unpickling."""
    path = Path(path)
    if expected_sha256:
        actual = sha256_file(path)
        if actual.lower() != expected_sha256.lower():
            raise ValueError(
                f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}. "
                "Refusing to unpickle a file that differs from the team's recorded copy."
            )
    else:
        log.warning("No expected SHA-256 given for %s: unpickling unverified file", path)
    df = read_pickle_with_aliases(path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{path} did not contain a DataFrame (got {type(df).__name__})")
    return df


def _flat(x) -> str | None:
    """Turn nested arrays like [['Edge-Ring']] into 'Edge-Ring'; empty or missing -> None."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    a = np.asarray(x, dtype=object).ravel()
    return str(a[0]) if a.size else None


__all__ = ["load_raw", "read_pickle_with_aliases", "sha256_file", "_flat"]

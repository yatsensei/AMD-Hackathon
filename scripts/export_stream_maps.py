"""Bundle the raw wafer maps for a stream into dashboard/data/stream_maps.npz.

The dashboard reads maps from this pack first, so a demo machine needs neither the
2 GB pickle nor `prepare_data.py`. Run once after (re)building a stream:

    python scripts/export_stream_maps.py --stream streams/stream_val.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wafer.data.access import get_wafer_map
from wafer.env import REPO_ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stream", default=str(REPO_ROOT / "streams" / "stream_val.csv"))
    ap.add_argument("--out", default=str(REPO_ROOT / "dashboard" / "data" / "stream_maps.npz"))
    a = ap.parse_args()

    uids = pd.read_csv(a.stream)["uid"].drop_duplicates().astype(int).tolist()
    maps = [np.asarray(get_wafer_map(u), dtype=np.uint8) for u in uids]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, uid=np.array(uids, dtype=np.int64),
                        shape=np.array([m.shape for m in maps], dtype=np.int32),
                        flat=np.concatenate([m.ravel() for m in maps]))
    print(f"{len(uids)} maps -> {a.out} ({Path(a.out).stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()

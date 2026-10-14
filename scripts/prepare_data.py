#!/usr/bin/env python
"""Phase 1 pipeline entry point. See wafer/data/pipeline.py and spec Section 3."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wafer.data.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()

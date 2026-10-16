"""Pattern -> likely cause, machine and action (the agent's `lookup_root_cause` tool)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from wafer.constants import CLASSES

ROOT_CAUSES = Path(__file__).with_name("root_causes.json")


@lru_cache(maxsize=1)
def load_root_causes(path: str | Path = ROOT_CAUSES) -> dict:
    """{pattern: {cause, machine, tool_family, recommended_action, source}} for all 9 classes."""
    table = {k: v for k, v in json.loads(Path(path).read_text()).items() if not k.startswith("_")}
    missing, extra = set(CLASSES) - set(table), set(table) - set(CLASSES)
    if missing or extra:
        raise ValueError(f"root_causes.json must cover exactly wafer.constants.CLASSES: "
                         f"missing {sorted(missing)}, unknown {sorted(extra)}")
    return table


def lookup_root_cause(pattern: str) -> dict:
    """Root-cause entry for one pattern; raises ValueError on an unknown pattern."""
    table = load_root_causes()
    if pattern not in table:
        raise ValueError(f"unknown pattern {pattern!r}; expected one of {CLASSES}")
    return {"pattern": pattern, **table[pattern]}

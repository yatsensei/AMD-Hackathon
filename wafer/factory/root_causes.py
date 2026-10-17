"""Pattern -> likely cause, machine and action (the agent's `lookup_root_cause` tool)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from wafer.constants import CLASSES

ROOT_CAUSES = Path(__file__).with_name("root_causes.json")


def _source_warnings(source: str) -> list[str]:
    """Flags for missing / unverified citations so the agent can surface them."""
    s = (source or "").lower()
    warnings = []
    if "no citation" in s or "triage heuristic" in s:
        warnings.append("unverified_no_citation")
    if "not read in full" in s:
        warnings.append("source_not_read_in_full")
    if "to confirm" in s:
        warnings.append("source_needs_confirmation")
    return warnings


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
    """Root-cause entry for one pattern; raises ValueError on an unknown pattern.

    Results are triage possibilities, not diagnoses. `source` is preserved; 
    `source_warnings` flags missing or unverified citations.
    """
    table = load_root_causes()
    if pattern not in table:
        raise ValueError(f"unknown pattern {pattern!r}; expected one of {CLASSES}")
    entry = table[pattern]
    source = entry["source"]
    return {
        "pattern": pattern,
        **entry,
        "is_possibility": True,
        "source_warnings": _source_warnings(source),
        "disclaimer": ("This is a possible root cause for triage, not a confirmed diagnosis. "
                       "Cite the source and any source_warnings when explaining."),
    }

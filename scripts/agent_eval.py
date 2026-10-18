#!/usr/bin/env python
"""Ask the copilot a varied question set against the demo line and write a report.

    python scripts/agent_eval.py                      # needs AMD_API_KEY, AMD_BASE_URL, AMD_MODEL (.env)
    python scripts/agent_eval.py --as-of 2026-10-13T08:15:00 --out docs/results/agent_eval.md

Each group is one conversation, so follow-ups ("and ETCH-02?") test memory. The report lists,
per question, the tools the agent called, whether they include the expected ones, latency,
and the answer, for a person to read. Nothing here grades the answer text automatically.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wafer.agent import WaferAgent  # noqa: E402
from wafer.env import REPO_ROOT  # noqa: E402

# (question, tools at least one of which should be called; empty = answerable from the snapshot)
CONVERSATIONS: list[list[tuple[str, set[str]]]] = [
    [("How is the line doing right now?", {"line_overview"}),
     ("Which tool is worst, and why?", {"line_overview", "check_lot_history", "defect_trend"}),
     ("When did that start?", {"defect_trend"}),
     ("Is it getting worse?", {"defect_trend"})],
    [("What is going on with ETCH-04?", {"check_lot_history"}),
     ("What should we check first, and what is at risk if we keep running?", {"lookup_root_cause", "estimate_impact"}),
     ("And ETCH-02?", {"check_lot_history", "line_overview", "defect_trend"})],
    [("Show me the wafers that were sent for review.", {"find_wafers"}),
     ("Why was the first one uncertain?", {"explain_wafer", "find_wafers"}),
     ("What else could it be?", {"explain_wafer"})],
    [("What does wafer 6 of LOT-014 look like to the model?", {"find_wafers", "explain_wafer"}),
     ("Where on the wafer is the defect?", {"explain_wafer"})],
    [("What is a Donut pattern?", {"describe_pattern"}),
     ("How is it different from Edge-Ring?", {"describe_pattern"}),
     ("Have we seen either on this line?", {"line_overview", "find_wafers", "check_lot_history"})],
    [("How accurate is the model?", {"model_performance"}),
     ("Can I trust it on Scratch?", {"model_performance"}),
     ("Why do you quote macro-F1 instead of accuracy?", set())],
    [("Which lots are held, and who decided?", {"get_decisions"}),
     ("What would it cost if we let two more lots run on ETCH-04?", {"estimate_impact"})],
    [("What's the weather in Sydney?", set())],
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stream", default=str(REPO_ROOT / "streams" / "stream_val_ens3.csv"))
    p.add_argument("--as-of", default="2026-10-13T08:15:00", help="line time (default: after the ETCH-04 alert)")
    p.add_argument("--out", default=str(REPO_ROOT / "docs" / "results" / "agent_eval.md"))
    a = p.parse_args(argv)

    df = pd.read_csv(a.stream)
    agent = WaferAgent.from_env(df, a.as_of, decisions=lambda: [], verify_model=True)
    lines = [f"# Copilot evaluation\n\nModel `{agent.client.config.model}` · line time {a.as_of} · "
             f"stream `{Path(a.stream).name}`\n"]
    hits = total = 0
    for ci, conv in enumerate(CONVERSATIONS, 1):
        history: list[dict] = []
        lines.append(f"\n## Conversation {ci}\n")
        for q, expect in conv:
            t0 = time.perf_counter()
            out = agent.ask(q, history=history)
            secs = time.perf_counter() - t0
            tools = [t["tool"] for t in out.get("tool_trace", [])]
            ok = bool(expect & set(tools)) if expect else True
            hits += ok
            total += 1
            answer = (out.get("answer") or out.get("error") or "").strip()
            history.append({"question": q, "answer": answer})
            lines.append(f"**Q:** {q}  \n*tools:* {', '.join(tools) or 'none'} · "
                         f"*expected one of:* {', '.join(sorted(expect)) or 'none needed'} · "
                         f"{'ok' if ok else 'CHECK'} · {secs:.1f} s\n\n{answer}\n")
            print(f"[{'ok' if ok else 'CHECK'}] {secs:5.1f}s  {q}  ->  {', '.join(tools) or '-'}")
    lines.insert(1, f"\nTool routing as expected on {hits} of {total} questions.\n")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines))
    print(f"\n{hits}/{total} routed as expected; report -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

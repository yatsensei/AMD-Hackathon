#!/usr/bin/env python
"""Spec 9.1 latency budget for a model folder: classify(), batch 1, 4 CPU threads, 10 warm-ups,
mean over 200 calls. Budget: 50 ms per wafer."""
import argparse
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wafer.bench import latency_ms  # noqa: E402
from wafer.infer import load_model  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", default="models/v0")
p.add_argument("--budget-ms", type=float, default=50.0)
a = p.parse_args()
ms = latency_ms(load_model(a.model, "cpu").classify)
print(f"{a.model}: {ms:.2f} ms per wafer on {platform.processor() or platform.machine()} "
      f"({'OK' if ms < a.budget_ms else 'OVER BUDGET'}, budget {a.budget_ms} ms)")
sys.exit(0 if ms < a.budget_ms else 1)

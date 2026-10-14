"""CPU latency for the served config (spec 9.1 reference: 4 threads, batch 1, 10 warm-ups)."""
from __future__ import annotations

import time

import numpy as np


def demo_wafer(h: int = 52, w: int = 52, seed: int = 0) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot((yy - (h - 1) / 2) / (h / 2), (xx - (w - 1) / 2) / (w / 2))
    m = (r <= 1).astype(np.uint8)
    rng = np.random.default_rng(seed)
    m[(m == 1) & (rng.random((h, w)) < 0.03)] = 2
    return m


def latency_ms(classify, n: int = 200, warmup: int = 10, threads: int = 4, wafer=None) -> float:
    import torch
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    m = demo_wafer() if wafer is None else wafer
    try:
        for _ in range(warmup):
            classify(m)
        t = time.perf_counter()
        for _ in range(n):
            classify(m)
        return (time.perf_counter() - t) / n * 1000
    finally:
        torch.set_num_threads(prev)

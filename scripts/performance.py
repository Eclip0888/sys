"""Measurements for Table 8 (Algorithm Evaluation Summary): time, memory, scalability, CPU.

    python scripts/performance.py
"""
import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402

from app import engine  # noqa: E402
from sim.simulation import Config, run_study, synthetic_pool  # noqa: E402

p = engine.Params()
feats = {"fkgl": 12.0, "len": 5.3, "tech": 0.25, "level": 2}
rng = np.random.default_rng(0)

print("## Time efficiency: live engine time per response (ms), by length of the user's history")
for n in (1, 10, 50, 100):
    hist = [{"x": int(rng.random() < 0.6), "b": float(rng.normal()), "w": 1.0} for _ in range(n - 1)]
    t0 = time.perf_counter()
    for _ in range(200):
        engine.evaluate(hist, feats, True, 10.0, 40, p)
    print(f"history {n:>3}: {1000 * (time.perf_counter() - t0) / 200:.3f} ms")

print("\n## Space efficiency and scalability: simulation (5 replications, 2 calibration sets)")
try:
    import psutil
    proc = psutil.Process()
except ImportError:
    proc = None
for users in (100, 1000, 10000):
    tracemalloc.start()
    t0, c0 = time.perf_counter(), time.process_time()
    run_study(synthetic_pool(), Config(replications=5, users=users, calibration_sets=2))
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss = f", process RAM {proc.memory_info().rss / 2**20:.0f} MB" if proc else ""
    print(f"users {users:>6}: {wall:.1f} s wall, CPU {100 * cpu / wall:.0f}%, peak Python memory {peak / 2**20:.1f} MB{rss}")

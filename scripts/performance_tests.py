"""Computational performance tests for Chapter 4, Table 3 (Research Question 7).

    python scripts/performance_tests.py                                   # synthetic 150-user study
    python scripts/performance_tests.py --responses responses.csv --users users.csv

Writes results/table3_performance.md, results/table3_performance.json and
results/table3_scalability.csv.
"""
import argparse
import json
import os
import platform
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import performance  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses")
    ap.add_argument("--users")
    ap.add_argument("--sizes", default="50,100,200,300,500", help="numbers of users for the scalability test")
    ap.add_argument("--repeats", type=int, default=3, help="cross-validation repeats for the robustness test")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    responses = users = None
    if a.responses and a.users:
        responses, users = pd.read_csv(a.responses), pd.read_csv(a.users)
    sizes = tuple(int(s) for s in a.sizes.split(","))
    r = performance.run(responses, users, sizes=sizes, n_repeats=a.repeats)
    os.makedirs(a.outdir, exist_ok=True)
    r["scalability"]["table"].to_csv(os.path.join(a.outdir, "table3_scalability.csv"), index=False)
    md = performance.report_markdown(r)
    md += f"\nMachine: {platform.platform()}, Python {platform.python_version()}, {os.cpu_count()} CPU threads, no GPU.\n"
    with open(os.path.join(a.outdir, "table3_performance.md"), "w") as fh:
        fh.write(md)
    out = {**r, "scalability": {**r["scalability"], "table": r["scalability"]["table"].to_dict("records")}}
    with open(os.path.join(a.outdir, "table3_performance.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(md)

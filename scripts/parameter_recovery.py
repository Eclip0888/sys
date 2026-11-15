"""Parameter recovery study for Chapter 4, Table 15.

    python scripts/parameter_recovery.py                 # 50 datasets x 150 users (about 1 minute)
    python scripts/parameter_recovery.py --datasets 10   # quicker check

Writes results/table15_parameter_recovery.csv (the table), results/parameter_recovery_estimates.csv
(every estimate, for the appendix) and results/table15_parameter_recovery.md.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import recovery  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=int, default=50)
    ap.add_argument("--users", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    est, table = recovery.run(n_datasets=a.datasets, n_users=a.users, seed=a.seed)
    os.makedirs(a.outdir, exist_ok=True)
    table.to_csv(os.path.join(a.outdir, "table15_parameter_recovery.csv"), index=False)
    est.to_csv(os.path.join(a.outdir, "parameter_recovery_estimates.csv"), index=False)
    md = recovery.report_markdown(table, a.users)
    with open(os.path.join(a.outdir, "table15_parameter_recovery.md"), "w") as fh:
        fh.write(md)
    print(md)

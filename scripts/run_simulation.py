"""Run the simulation study (H1-H4) from your computer.

    python scripts/run_simulation.py                     # thesis design: 100 replications, 200 users
    python scripts/run_simulation.py --quick             # 10 replications, to check that it runs
    python scripts/run_simulation.py --pool synthetic    # without the prompt-set answers

Writes results/simulation_report.md and the CSV files for Tables 4 to 7 and 9.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sim.simulation import Config, report_markdown, run_study, save, synthetic_pool  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--replications", type=int, default=100)
ap.add_argument("--users", type=int, default=200)
ap.add_argument("--calibration-sets", type=int, default=20)
ap.add_argument("--pool", choices=["db", "synthetic"], default="db")
ap.add_argument("--batch", default=None)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default="results")
a = ap.parse_args()
if a.quick:
    a.replications, a.calibration_sets = 10, 3

pool, source = None, "synthetic"
if a.pool == "db":
    from app import create_app, jobs
    with create_app().app_context():
        pool = jobs.pool_from_db(a.batch)
    if pool is not None:
        source = f"{len(pool)} Gemini answers from the prompt set"
    else:
        print("Fewer than 30 prompt-set answers in the database: using the synthetic pool.")
if pool is None:
    pool = synthetic_pool()
cfg = Config(replications=a.replications, users=a.users, calibration_sets=a.calibration_sets, seed=a.seed,
             pool_source=source)
rep = run_study(pool, cfg, progress=lambda d, t: print(f"\r{d}/{t}", end="", flush=True))
print()
save(rep, a.out)
print(report_markdown(rep))

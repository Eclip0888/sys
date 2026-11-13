"""Run the research evaluation (H1-H5 and baselines) on the data the app collected.

    python scripts/run_evaluation.py --responses data/responses.csv --users data/users.csv
    python scripts/run_evaluation.py --demo            # synthetic 150-user study (single sitting, 24 checks)

Writes results/evaluation.json and results/evaluation_report.md.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import evaluate                    # noqa: E402
from engine.simulate import simulate_study     # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses")
    ap.add_argument("--users")
    ap.add_argument("--demo", action="store_true", help="use a synthetic 150-user dataset")
    ap.add_argument("--irt-model", default="2pl", choices=["2pl", "3pl"])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    if a.demo:
        responses, users = simulate_study(n_users=150, seed=7)
    elif a.responses and a.users:
        responses, users = pd.read_csv(a.responses), pd.read_csv(a.users)
    else:
        ap.error("give --responses and --users, or --demo")
    r = evaluate.run(responses, users, irt_model=a.irt_model, n_repeats=a.repeats)
    os.makedirs(a.outdir, exist_ok=True)
    evaluate.save(r, os.path.join(a.outdir, "evaluation.json"), os.path.join(a.outdir, "evaluation_report.md"))
    # Chapter 4 tables for the algorithm tests
    pd.DataFrame(r["H5_ablation"]["table"]).to_csv(os.path.join(a.outdir, "table16_ablation.csv"), index=False)
    h1, h2, bf, ra = r["H1_irt"], r["H2_bkt"], r["baseline_bkt_forgetting"], r["baseline_irt_rasch"]
    pd.DataFrame([
        {"model": "Rasch (1PL) IRT", "family": "IRT", "loglik": ra["loglik"], "aic": ra["aic"], "bic": ra["bic"]},
        {"model": "Standard 2PL IRT", "family": "IRT", "loglik": h1["standard"]["loglik"], "aic": h1["standard"]["aic"], "bic": h1["standard"]["bic"]},
        {"model": "Augmented 2PL IRT", "family": "IRT", "loglik": h1["augmented"]["loglik"], "aic": h1["augmented"]["aic"], "bic": h1["augmented"]["bic"]},
        {"model": "Standard BKT", "family": "BKT", "auc": h2["standard"]["auc"], "rmse": h2["standard"]["rmse"]},
        {"model": "BKT with forgetting", "family": "BKT", "auc": bf["auc"], "rmse": bf["rmse"]},
        {"model": "Augmented BKT", "family": "BKT", "auc": h2["augmented"]["auc"], "rmse": h2["augmented"]["rmse"]},
    ]).to_csv(os.path.join(a.outdir, "table18_baselines.csv"), index=False)
    print(evaluate.report_markdown(r))

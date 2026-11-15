"""Parameter recovery study (Chapter 4, Table 15).

Synthetic datasets are generated with KNOWN values of the augmented parameters, the engine
fits them, and the estimates are compared with the truth:

    bias      mean(estimate) - true value
    RMSE      sqrt(mean((estimate - true value)^2))
    coverage  share of datasets whose 95% interval (estimate +/- 1.96 SE) contains the true value

Recovery is accepted when |bias| < 10% of the true value (the criterion set in Chapter 3).
Coverage below 95% is reported as a sign that the standard errors are optimistic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bkt import BKT, LEVELS
from .irt import AugmentedIRT
from .simulate import simulate_bkt, simulate_irt

TRUE_LAMBDA = 0.8
TRUE_T = (0.05, 0.12, 0.20)          # P(T | low), P(T | medium), P(T | high)
BIAS_LIMIT = 0.10                     # |bias| must be below 10% of the true value


def _pooled_bkt(df: pd.DataFrame) -> pd.DataFrame:
    """Fit one set of trust-level rates over all skills: each user-skill sequence is kept
    separate, so pooling only increases the number of sequences, not their length."""
    out = df.copy()
    out["user_id"] = out["user_id"].astype(str) + "_" + out["skill"].astype(str)
    out["skill"] = "ALL"
    return out


def one_dataset(seed: int, n_users=150, n_items=24, steps_per_skill=8,
                lam=TRUE_LAMBDA, T=TRUE_T) -> list[dict]:
    """Simulate one study-sized dataset, fit both augmented models, return one row per parameter."""
    rows = []
    d = simulate_irt(n_users=n_users, n_items=n_items, lam=lam, seed=seed)
    r = AugmentedIRT("2pl", augmented=True).fit(d["X"], d["conf"])
    rows.append({"dataset": seed, "parameter": "lambda", "true": lam, "estimate": r.lam, "se": r.lam_se})

    df = simulate_bkt(n_users=n_users, steps_per_skill=steps_per_skill, T=T, seed=10_000 + seed)
    m = BKT(augmented=True).fit(_pooled_bkt(df))
    est, se = m.params["ALL"].T, m.t_standard_errors("ALL")
    for k, lvl in enumerate(LEVELS):
        rows.append({"dataset": seed, "parameter": f"T_{lvl}", "true": T[k], "estimate": float(est[k]),
                     "se": float(se[k])})
    return rows


def run(n_datasets=50, n_users=150, n_items=24, steps_per_skill=8, seed=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-dataset estimates, Table 15 summary)."""
    rows = []
    for k in range(n_datasets):
        rows += one_dataset(seed + k, n_users, n_items, steps_per_skill)
    est = pd.DataFrame(rows)
    est["covered"] = (est["estimate"] - 1.96 * est["se"] <= est["true"]) & (est["true"] <= est["estimate"] + 1.96 * est["se"])
    est["error"] = est["estimate"] - est["true"]
    g = est.groupby("parameter", sort=False)
    table = pd.DataFrame({
        "true_value": g["true"].first(),
        "mean_estimate": g["estimate"].mean(),
        "bias": g["error"].mean(),
        "rmse": g["error"].apply(lambda e: float(np.sqrt(np.mean(e ** 2)))),
        "coverage_pct": g["covered"].mean() * 100,
        "n_datasets": g.size(),
    })
    table["bias_pct_of_true"] = table["bias"].abs() / table["true_value"] * 100
    table["accepted"] = table["bias_pct_of_true"] < BIAS_LIMIT * 100
    labels = {"lambda": "λ (metacognitive weight)", "T_low": "P(T | low trust)",
              "T_medium": "P(T | medium trust)", "T_high": "P(T | high trust)"}
    table.insert(0, "label", [labels[p] for p in table.index])
    return est, table.reset_index()


def report_markdown(table: pd.DataFrame, n_users: int) -> str:
    lines = ["# Table 15 – Recovery of the augmented parameters on synthetic data", "",
             f"{int(table['n_datasets'].iloc[0])} synthetic datasets, {n_users} users each "
             "(24 IRT items; 8 BKT steps per explanation type).", "",
             "| Parameter | True value | Mean estimate | Bias | RMSE | 95% CI coverage | |bias| < 10% of true |",
             "|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {r['label']} | {r['true_value']:.2f} | {r['mean_estimate']:.3f} | {r['bias']:+.3f} | "
                     f"{r['rmse']:.3f} | {r['coverage_pct']:.0f}% | {'yes' if r['accepted'] else 'no'} |")
    return "\n".join(lines) + "\n"

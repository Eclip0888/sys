"""ASSISTments 2009-2010 benchmark for the standard IRT and BKT models.

ASSISTments has no confidence or trust ratings, so it cannot test the augmentations.  It is
used, as Chapter 3 states, to check that the engine's STANDARD models behave as expected on a
large public dataset (Rasch vs 2PL IRT; standard BKT vs BKT with forgetting), and to time the
engine on real data of increasing size.

Expected file: the "skill builder" CSV (skill_builder_data.csv or pyBKT's as.csv) with at least
the columns order_id, user_id, problem_id, skill_name, correct (and optionally original).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import bkt as bkt_mod
from . import irt as irt_mod

USECOLS = ["order_id", "user_id", "problem_id", "skill_name", "correct", "original"]


def load(path: str) -> pd.DataFrame:
    """Read and clean the file: main problems only, one row per answer, 0/1 correctness."""
    head = pd.read_csv(path, nrows=0, encoding="latin-1").columns
    df = pd.read_csv(path, usecols=[c for c in USECOLS if c in head], encoding="latin-1", low_memory=False)
    if "original" in df:
        df = df[df["original"] == 1]                 # drop scaffolding sub-questions
    df = df.dropna(subset=["skill_name", "correct"])
    df = df[df["correct"].isin([0, 1])]
    df = df.drop_duplicates(["order_id", "skill_name"])
    df["correct"] = df["correct"].astype(int)
    return df.reset_index(drop=True)


def bkt_data(df: pd.DataFrame, n_skills=10, min_opportunities=3) -> pd.DataFrame:
    """The n most-practised skills, users with at least `min_opportunities` answers per skill.
    Trust is set to 3 (medium) because ASSISTments has no trust ratings."""
    top = df["skill_name"].value_counts().index[:n_skills]
    d = df[df["skill_name"].isin(top)].drop_duplicates(["order_id"])
    n = d.groupby(["user_id", "skill_name"])["order_id"].transform("size")
    d = d[n >= min_opportunities]
    return pd.DataFrame({"user_id": d["user_id"].astype(str), "skill": d["skill_name"],
                         "order": d["order_id"], "correct": d["correct"], "trust": 3}).reset_index(drop=True)


def irt_matrix(df: pd.DataFrame, n_items=50, min_answers=10) -> pd.DataFrame:
    """Users x problems (first attempt, NaN = not answered) for the n most-answered problems."""
    d = df.sort_values("order_id").drop_duplicates(["user_id", "problem_id"])
    top = d["problem_id"].value_counts().index[:n_items]
    X = d[d["problem_id"].isin(top)].pivot(index="user_id", columns="problem_id", values="correct")
    return X[X.notna().sum(axis=1) >= min_answers]


def benchmark(df: pd.DataFrame, n_items=50, n_skills=10, n_repeats=2, fractions=(0.25, 0.5, 1.0), seed=0) -> dict:
    out = {"n_rows": int(len(df)), "n_users_total": int(df["user_id"].nunique()),
           "n_skills_total": int(df["skill_name"].nunique())}

    # ---------------- IRT: Rasch vs standard 2PL
    X = irt_matrix(df, n_items=n_items)
    Xn = X.to_numpy(float)
    t0 = time.perf_counter()
    m2 = irt_mod.AugmentedIRT("2pl", augmented=False)
    r2 = m2.fit(Xn)
    t_2pl = time.perf_counter() - t0
    r1 = irt_mod.AugmentedIRT("1pl", augmented=False).fit(Xn)
    lr = irt_mod.likelihood_ratio_test(r1, r2)
    out["irt"] = {"n_users": int(X.shape[0]), "n_items": int(X.shape[1]), "n_responses": int(X.notna().sum().sum()),
                  "rasch": {"loglik": r1.loglik, "aic": r1.aic, "bic": r1.bic, "n_params": r1.n_params},
                  "2pl": {"loglik": r2.loglik, "aic": r2.aic, "bic": r2.bic, "n_params": r2.n_params,
                          "rmse": m2.rmse()},
                  "lr_test_2pl_vs_rasch": lr, "fit_seconds_2pl": t_2pl,
                  "mean_discrimination": float(np.mean(r2.a)), "sd_difficulty": float(np.std(r2.b))}

    # ---------------- BKT: standard vs forgetting (cross-validated by user)
    B = bkt_data(df, n_skills=n_skills)
    cv = bkt_mod.cross_validate(B, n_splits=5, n_repeats=n_repeats, seed=seed,
                                variants=(("standard", False, False), ("forgetting", False, True)))
    summ = cv.groupby("model")[["auc", "rmse", "accuracy"]].mean()
    out["bkt"] = {"n_users": int(B["user_id"].nunique()), "n_skills": int(B["skill"].nunique()),
                  "n_responses": int(len(B)), "skills": sorted(B["skill"].unique().tolist()),
                  "standard": summ.loc["standard"].to_dict(), "forgetting": summ.loc["forgetting"].to_dict()}

    # ---------------- scalability on real data
    rng = np.random.default_rng(seed)
    users = B["user_id"].unique()
    rows = []
    for f in fractions:
        keep = rng.choice(users, size=max(2, int(len(users) * f)), replace=False)
        sub = B[B["user_id"].isin(keep)]
        t0 = time.perf_counter(); bkt_mod.BKT(augmented=False).fit(sub); t_b = time.perf_counter() - t0
        xs = Xn[: max(2, int(len(Xn) * f))]
        t0 = time.perf_counter(); irt_mod.AugmentedIRT("2pl", augmented=False).fit(xs); t_i = time.perf_counter() - t0
        rows.append({"fraction": f, "bkt_responses": int(len(sub)), "bkt_seconds": t_b,
                     "irt_responses": int(np.isfinite(xs).sum()), "irt_seconds": t_i})
    out["scalability"] = rows
    return out


def report_markdown(r: dict) -> str:
    i, b = r["irt"], r["bkt"]
    lines = ["# ASSISTments 2009–2010 benchmark (standard models)", "",
             f"File: {r['n_rows']:,} answers from {r['n_users_total']:,} students on {r['n_skills_total']} skills "
             "(main problems only).", "",
             f"## IRT – {i['n_users']:,} students × {i['n_items']} most-answered problems ({i['n_responses']:,} answers)", "",
             "| Model | Log-likelihood | AIC | BIC | Parameters |", "|---|---|---|---|---|",
             f"| Rasch (1PL) | {i['rasch']['loglik']:.1f} | {i['rasch']['aic']:.1f} | {i['rasch']['bic']:.1f} | {i['rasch']['n_params']} |",
             f"| Standard 2PL | {i['2pl']['loglik']:.1f} | {i['2pl']['aic']:.1f} | {i['2pl']['bic']:.1f} | {i['2pl']['n_params']} |",
             "", f"2PL vs Rasch: χ²({i['lr_test_2pl_vs_rasch']['df']}) = {i['lr_test_2pl_vs_rasch']['lr']:.1f}, "
                 f"p = {i['lr_test_2pl_vs_rasch']['p']:.4g}; 2PL RMSE = {i['2pl']['rmse']:.3f}; "
                 f"fit time {i['fit_seconds_2pl']:.1f} s.", "",
             f"## BKT – {b['n_users']:,} students, {b['n_skills']} most-practised skills ({b['n_responses']:,} answers), "
             "5-fold cross-validation by student", "",
             "| Model | AUC | RMSE | Accuracy |", "|---|---|---|---|",
             f"| Standard BKT | {b['standard']['auc']:.3f} | {b['standard']['rmse']:.3f} | {b['standard']['accuracy']:.3f} |",
             f"| BKT with forgetting | {b['forgetting']['auc']:.3f} | {b['forgetting']['rmse']:.3f} | {b['forgetting']['accuracy']:.3f} |",
             "", "## Runtime on real data", "",
             "| Share of students | BKT answers | BKT fit (s) | IRT answers | IRT fit (s) |", "|---|---|---|---|---|"]
    for s in r["scalability"]:
        lines.append(f"| {s['fraction']:.0%} | {s['bkt_responses']:,} | {s['bkt_seconds']:.2f} | "
                     f"{s['irt_responses']:,} | {s['irt_seconds']:.2f} |")
    return "\n".join(lines) + "\n"

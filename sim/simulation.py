"""Simulation study of the real-time comprehension-evaluation algorithm (Chapter 3, Simulation Design).

Aims        how accurately the algorithm identifies true understanding and recovers ability, and how much
            each augmentation contributes.
Data        simulated users with true ability theta ~ N(0, 1) and rapid-guessing propensity pi ~ Beta(1, 9)
            (low) or Beta(2, 6) (high); 15 interactions each with answers drawn from a pool of AI answers;
            true difficulty b* = beta0* + beta* . z + e,  e ~ N(0, 0.3^2);  true understanding
            U ~ Bernoulli(1 / (1 + exp(-a* (theta - b*))));  a rapid guess (prob. pi) is a random option in
            about 1.5 s; otherwise correct with prob. 0.95 if U = 1 and 0.25 if U = 0, with a response time
            around the expected reading time 3 + L / 4 seconds.
Estimands   U for every response, theta for every user.
Methods     raw correctness, fixed-difficulty IRT, Elo, complexity only, response time only, full algorithm.
Performance accuracy, precision, recall, F1, AUC (U); RMSE, bias, r (theta); runtime.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

NODES = np.linspace(-4, 4, 41)
LOGPRIOR = -0.5 * NODES ** 2
LOGPRIOR -= np.log(np.exp(LOGPRIOR).sum())
C_GUESS, SLIP = 0.25, 0.05
TRUE = {"beta0": 0.0, "beta": np.array([0.5, 0.3, 0.4, 0.5]), "a": 1.2, "eps_sd": 0.3}
GUESSING = {"low": (1.0, 9.0), "high": (2.0, 6.0)}
METHODS = ("raw", "fixed", "elo", "complexity", "rt", "full")
METHOD_NAMES = {"raw": "Raw correctness", "fixed": "Fixed-difficulty IRT", "elo": "Elo rating",
                "complexity": "Complexity only", "rt": "Response time only", "full": "Full algorithm"}
OMEGA_MIN, ELO_K = 0.2, 0.4


# ------------------------------------------------------------------ answer pool
def synthetic_pool(n=200, seed=0) -> pd.DataFrame:
    """Stand-in pool used until the 200 real Gemini answers are available (prompt-set run)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "fkgl": np.clip(rng.normal(11, 3, n), 3, 22),
        "len": np.log(np.clip(rng.normal(180, 60, n), 40, 400)),
        "tech": np.clip(rng.normal(0.22, 0.08, n), 0.02, 0.6),
        "level": rng.choice([1, 2, 3], n, p=[0.4, 0.4, 0.2]),
        "qwords": np.clip(rng.normal(40, 10, n), 15, 90).round(),
    })


def standardize(pool: pd.DataFrame):
    X = pool[["fkgl", "len", "tech", "level"]].to_numpy(float)
    mean, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    return (X - mean) / sd, mean, sd


# ------------------------------------------------------------------ data generation
@dataclass
class Dataset:
    items: np.ndarray       # N x T item index into the pool
    U: np.ndarray           # N x T true understanding
    X: np.ndarray           # N x T correct tap
    t: np.ndarray           # N x T response time (s)
    tg: np.ndarray          # N x T rapid-guess threshold (s)
    rapid: np.ndarray       # N x T true rapid guess
    theta: np.ndarray       # N true ability
    b_true: np.ndarray      # J true difficulty


def generate(Z, qwords, n_users, n_steps, guessing, rng) -> Dataset:
    J = len(Z)
    b_true = TRUE["beta0"] + Z @ TRUE["beta"] + rng.normal(0, TRUE["eps_sd"], J)
    theta = rng.normal(0, 1, n_users)
    pi = rng.beta(*GUESSING[guessing], n_users)
    items = np.stack([rng.choice(J, n_steps, replace=False) for _ in range(n_users)])
    b = b_true[items]
    pU = 1 / (1 + np.exp(-TRUE["a"] * (theta[:, None] - b)))
    U = (rng.random(b.shape) < pU).astype(int)
    rapid = rng.random(b.shape) < pi[:, None]
    p_correct = np.where(U == 1, 1 - SLIP, C_GUESS)
    X = np.where(rapid, rng.random(b.shape) < C_GUESS, rng.random(b.shape) < p_correct).astype(int)
    texp = 3 + qwords[items] / 4
    t = np.where(rapid, rng.lognormal(np.log(1.5), 0.3, b.shape), rng.lognormal(np.log(texp), 0.35))
    tg = np.maximum(2.0, 0.10 * texp)
    return Dataset(items, U, X, t, tg, rapid, theta, b_true)


# ------------------------------------------------------------------ model pieces
def p_und(theta, b, a):
    return 1 / (1 + np.exp(-a * (theta - b)))


def p_cor(theta, b, a):
    return C_GUESS + (1 - C_GUESS - SLIP) * p_und(theta, b, a)


def weights(d: Dataset, use_rt: bool):
    return np.where(d.t < d.tg, OMEGA_MIN, 1.0) if use_rt else np.ones_like(d.t)


def item_difficulty(Z, params, use_features: bool):
    return params["beta0"] + (Z @ params["beta"] if use_features else np.zeros(len(Z)))


def calibrate(d: Dataset, Z, use_features: bool, use_rt: bool) -> dict:
    """Marginal maximum likelihood (41 quadrature nodes) of beta0, beta (LLTM) or beta0 only, and a."""
    w = weights(d, use_rt)
    k = Z.shape[1] if use_features else 0

    def unpack(v):
        return {"beta0": v[0], "beta": np.array(v[1:1 + k]) if k else np.zeros(Z.shape[1]), "a": np.exp(v[-1])}

    def nll(v):
        prm = unpack(v)
        b = item_difficulty(Z, prm, use_features)[d.items]                      # N x T
        P = np.clip(p_cor(NODES[None, None, :], b[..., None], prm["a"]), 1e-9, 1 - 1e-9)
        ll = np.where(d.X[..., None] == 1, np.log(P), np.log(1 - P)) * w[..., None]
        L = ll.sum(1) + LOGPRIOR[None, :]                                        # N x Q
        m = L.max(1, keepdims=True)
        return -(m[:, 0] + np.log(np.exp(L - m).sum(1))).sum()

    v0 = np.r_[0.0, np.zeros(k), 0.0]
    r = optimize.minimize(nll, v0, method="L-BFGS-B")
    out = unpack(r.x)
    out["converged"] = bool(r.success)
    return out


# ------------------------------------------------------------------ scoring methods
def score_irt(d: Dataset, b_items: np.ndarray, a: float, w: np.ndarray):
    """Sequential weighted EAP for every user at once. Returns C (N x T) and theta-hat after each step."""
    N, T = d.X.shape
    logpost = np.tile(LOGPRIOR, (N, 1))
    C = np.zeros((N, T))
    th = np.zeros((N, T))
    for s in range(T):
        b = b_items[d.items[:, s]][:, None]
        x, ws = d.X[:, s][:, None], w[:, s][:, None]
        post = np.exp(logpost - logpost.max(1, keepdims=True))
        post /= post.sum(1, keepdims=True)
        pi = p_und(NODES[None, :], b, a)
        l1 = np.where(x == 1, 1 - SLIP, SLIP) ** ws
        l0 = np.where(x == 1, C_GUESS, 1 - C_GUESS) ** ws
        num = (post * pi * l1).sum(1)
        C[:, s] = num / (num + (post * (1 - pi) * l0).sum(1))
        P = np.clip(p_cor(NODES[None, :], b, a), 1e-9, 1 - 1e-9)
        logpost = logpost + ws * np.where(x == 1, np.log(P), np.log(1 - P))
        post = np.exp(logpost - logpost.max(1, keepdims=True))
        post /= post.sum(1, keepdims=True)
        th[:, s] = (post * NODES[None, :]).sum(1)
    return C, th


def score_elo(d: Dataset):
    N, T = d.X.shape
    theta = np.zeros(N)
    C, th = np.zeros((N, T)), np.zeros((N, T))
    for s in range(T):
        p = 1 / (1 + np.exp(-theta))
        theta = theta + ELO_K * (d.X[:, s] - p)
        C[:, s] = 1 / (1 + np.exp(-theta))
        th[:, s] = theta
    return C, th


def run_methods(d: Dataset, Z, cal: dict):
    out = {"raw": (d.X.astype(float), None)}
    out["elo"] = score_elo(d)
    spec = {"fixed": (False, False), "complexity": (True, False), "rt": (False, True), "full": (True, True)}
    for m, (feat, rt) in spec.items():
        prm = cal[m]
        out[m] = score_irt(d, item_difficulty(Z, prm, feat), prm["a"], weights(d, rt))
    return out


def classification(U, score):
    y, s = U.ravel(), score.ravel()
    pred = (s >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "auc": roc_auc_score(y, s) if len(np.unique(s)) > 1 else 0.5}


def recovery(theta, th, steps=(5, 10, 15)):
    out = {}
    for k in steps:
        e = th[:, k - 1] - theta
        out[f"rmse_{k}"] = float(np.sqrt((e ** 2).mean()))
        out[f"bias_{k}"] = float(e.mean())
        out[f"r_{k}"] = float(np.corrcoef(th[:, k - 1], theta)[0, 1])
    return out


# ------------------------------------------------------------------ statistics
def paired(a, b):
    """Paired comparison of a vs b across replications (a should be larger): t-test or Wilcoxon."""
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    normal = len(d) < 3 or np.allclose(d, d[0]) or stats.shapiro(d).pvalue >= 0.05
    if normal:
        res = stats.ttest_rel(a, b, alternative="greater")
        test, stat, p = "paired t", float(res.statistic), float(res.pvalue)
    else:
        res = stats.wilcoxon(a, b, alternative="greater")
        test, stat, p = "Wilcoxon", float(res.statistic), float(res.pvalue)
    dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0
    return {"test": test, "stat": stat, "p": p, "dz": dz, "mean_diff": float(d.mean())}


# ------------------------------------------------------------------ the study
@dataclass
class Config:
    replications: int = 100
    users: int = 200
    steps: int = 15
    calibration_users: int = 500
    calibration_sets: int = 20
    seed: int = 2026
    alpha: float = 0.05
    pool_source: str = "synthetic"
    extra: dict = field(default_factory=dict)


def run_study(pool: pd.DataFrame, cfg: Config, progress=None) -> dict:
    t_start = time.time()
    Z, mean, sd = standardize(pool)
    qwords = pool["qwords"].to_numpy(float)
    rng = np.random.default_rng(cfg.seed)
    results, recov_rows, calib_rows = [], [], []
    total = 2 * (cfg.calibration_sets + cfg.replications)
    done = 0
    timing = {"score_seconds": 0.0, "responses_scored": 0}
    for cond in ("low", "high"):
        cals = []
        for c in range(cfg.calibration_sets):
            dc = generate(Z, qwords, cfg.calibration_users, cfg.steps, cond, rng)
            cal = {"fixed": calibrate(dc, Z, False, False), "complexity": calibrate(dc, Z, True, False),
                   "rt": calibrate(dc, Z, False, True), "full": calibrate(dc, Z, True, True)}
            cals.append(cal)
            f = cal["full"]
            calib_rows.append({"condition": cond, "set": c, "beta0": f["beta0"], **{f"beta{k + 1}": f["beta"][k] for k in range(4)},
                               "a": f["a"]})
            done += 1
            if progress:
                progress(done, total)
        for r in range(cfg.replications):
            d = generate(Z, qwords, cfg.users, cfg.steps, cond, rng)
            t0 = time.perf_counter()
            scores = run_methods(d, Z, cals[r % cfg.calibration_sets])
            timing["score_seconds"] += time.perf_counter() - t0
            timing["responses_scored"] += d.X.size * len(METHODS)
            for m, (C, th) in scores.items():
                row = {"condition": cond, "rep": r, "method": m, **classification(d.U, C)}
                if th is not None:
                    row.update(recovery(d.theta, th))
                results.append(row)
            pc = d.X.cumsum(1)[:, -1] / cfg.steps
            recov_rows.append({"condition": cond, "rep": r, "r_prop_15": float(np.corrcoef(pc, d.theta)[0, 1])})
            done += 1
            if progress:
                progress(done, total)
    res = pd.DataFrame(results)
    prop = pd.DataFrame(recov_rows)
    calib = pd.DataFrame(calib_rows)
    report = summarize(res, prop, calib, cfg)
    report["standardization"] = {"mean": mean.tolist(), "sd": sd.tolist()}
    report["runtime"] = {"total_seconds": round(time.time() - t_start, 1),
                         "ms_per_response": round(1000 * timing["score_seconds"] / max(timing["responses_scored"], 1), 5)}
    report["frames"] = {"results": res, "prop": prop, "calibration": calib}
    return report


def summarize(res, prop, calib, cfg: Config) -> dict:
    mean = res.groupby(["condition", "method"]).mean(numeric_only=True).reset_index()

    def vals(cond, method, metric):
        return res[(res.condition == cond) & (res.method == method)].sort_values("rep")[metric].to_numpy()

    tests = {}
    for cond in ("low", "high"):
        tests[f"H1_f1_{cond}"] = paired(vals(cond, "full", "f1"), vals(cond, "raw", "f1"))
        tests[f"H1_auc_{cond}"] = paired(vals(cond, "full", "auc"), vals(cond, "raw", "auc"))
        tests[f"H2_complexity_vs_fixed_{cond}"] = paired(vals(cond, "complexity", "auc"), vals(cond, "fixed", "auc"))
        tests[f"H2_full_vs_rt_{cond}"] = paired(vals(cond, "full", "auc"), vals(cond, "rt", "auc"))
    tests["H3_full_vs_complexity_high"] = paired(vals("high", "full", "auc"), vals("high", "complexity", "auc"))
    tests["H4_full_vs_elo_high"] = paired(-vals("high", "full", "rmse_15"), -vals("high", "elo", "rmse_15"))
    tests["H4_full_vs_fixed_high"] = paired(-vals("high", "full", "rmse_15"), -vals("high", "fixed", "rmse_15"))
    a = cfg.alpha
    ok = lambda *keys: all(tests[k]["p"] < a and tests[k]["mean_diff"] > 0 for k in keys)
    decisions = {
        "H1": ok("H1_f1_low", "H1_f1_high", "H1_auc_low", "H1_auc_high"),
        "H2": ok("H2_complexity_vs_fixed_low", "H2_complexity_vs_fixed_high", "H2_full_vs_rt_low", "H2_full_vs_rt_high"),
        "H3": ok("H3_full_vs_complexity_high"),
        "H4": ok("H4_full_vs_elo_high", "H4_full_vs_fixed_high"),
    }
    truth = {"beta0": TRUE["beta0"], **{f"beta{k + 1}": TRUE["beta"][k] for k in range(4)}, "a": TRUE["a"]}
    cal_tab = []
    for k, v in truth.items():
        est = calib[k].to_numpy()
        cal_tab.append({"parameter": k, "true": v, "mean_estimate": est.mean(), "bias": est.mean() - v,
                        "rmse": float(np.sqrt(((est - v) ** 2).mean()))})
    return {"config": cfg.__dict__, "mean": mean, "tests": tests, "decisions": decisions,
            "calibration_table": pd.DataFrame(cal_tab),
            "prop_r15_high": float(prop[prop.condition == "high"].r_prop_15.mean()),
            "prop_r15_low": float(prop[prop.condition == "low"].r_prop_15.mean())}


def report_markdown(rep: dict) -> str:
    m = rep["mean"]
    f = lambda x, d=3: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"
    get = lambda c, meth, k: m[(m.condition == c) & (m.method == meth)][k].iloc[0]
    L = ["# Simulation results", "",
         f"Replications: {rep['config']['replications']} per condition · users: {rep['config']['users']} · "
         f"interactions: {rep['config']['steps']} · calibration sets: {rep['config']['calibration_sets']} · "
         f"answer pool: {rep['config']['pool_source']} · runtime {rep['runtime']['total_seconds']} s", "",
         "## Table 4. Recovery of the calibrated parameters (full algorithm)", "",
         "| Parameter | True | Mean estimate | Bias | RMSE |", "|---|---|---|---|---|"]
    for _, r in rep["calibration_table"].iterrows():
        L.append(f"| {r.parameter} | {f(r.true, 2)} | {f(r.mean_estimate)} | {f(r.bias)} | {f(r.rmse)} |")
    L += ["", "## Table 5. Identification of true comprehension", "",
          "| Method | Guessing | Accuracy | Precision | Recall | F1 | AUC |", "|---|---|---|---|---|---|---|"]
    for c in ("low", "high"):
        for meth in ("raw", "fixed", "elo", "full"):
            L.append(f"| {METHOD_NAMES[meth]} | {c.title()} | " + " | ".join(f(get(c, meth, k)) for k in
                     ("accuracy", "precision", "recall", "f1", "auc")) + " |")
    L += ["", "## Table 6. Ablation (AUC)", "", "| Configuration | Low guessing | High guessing |", "|---|---|---|"]
    for meth in ("fixed", "complexity", "rt", "full"):
        L.append(f"| {METHOD_NAMES[meth]} | {f(get('low', meth, 'auc'))} | {f(get('high', meth, 'auc'))} |")
    L += ["", "## Table 7. Ability recovery (high guessing)", "", "| Method | RMSE (5) | RMSE (10) | RMSE (15) | r (15) |",
          "|---|---|---|---|---|", f"| Proportion correct | — | — | — | {f(rep['prop_r15_high'])} |"]
    for meth in ("elo", "fixed", "full"):
        L.append(f"| {METHOD_NAMES[meth]} | " + " | ".join(f(get("high", meth, k)) for k in ("rmse_5", "rmse_10", "rmse_15", "r_15")) + " |")
    L += ["", "## Hypothesis tests (one-sided, paired across replications)", "",
          "| Test | Test used | Statistic | p | dz | Mean difference |", "|---|---|---|---|---|---|"]
    for k, t in rep["tests"].items():
        L.append(f"| {k} | {t['test']} | {f(t['stat'], 2)} | {t['p']:.2g} | {f(t['dz'], 2)} | {f(t['mean_diff'], 4)} |")
    L += ["", "| Hypothesis | Decision |", "|---|---|"]
    for k, v in rep["decisions"].items():
        L.append(f"| {k} | {'Supported' if v else 'Not supported'} |")
    L += ["", f"Evaluation time: {rep['runtime']['ms_per_response']} ms per response (vectorized simulation scoring)."]
    return "\n".join(L)


def save(rep: dict, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    for name, df in rep["frames"].items():
        df.to_csv(os.path.join(outdir, f"sim_{name}.csv"), index=False)
    rep["mean"].to_csv(os.path.join(outdir, "sim_means.csv"), index=False)
    rep["calibration_table"].to_csv(os.path.join(outdir, "table4_calibration.csv"), index=False)
    with open(os.path.join(outdir, "simulation_report.md"), "w", encoding="utf-8") as fh:
        fh.write(report_markdown(rep))
    slim = {k: v for k, v in rep.items() if k not in ("frames", "mean", "calibration_table")}
    with open(os.path.join(outdir, "simulation_summary.json"), "w") as fh:
        json.dump(slim, fh, indent=1, default=float)


def calibrated_params(rep: dict) -> dict:
    """Mean calibrated full-algorithm parameters (high guessing), for use in the live chatbot."""
    cal = rep["frames"]["calibration"]
    c = cal[cal.condition == "high"].mean(numeric_only=True)
    st = rep["standardization"]
    return {"beta0": float(c.beta0), "beta": [float(c[f"beta{k}"]) for k in range(1, 5)], "a": float(c.a),
            "mean": st["mean"], "sd": st["sd"], "source": f"simulation calibration ({rep['config']['pool_source']} pool)"}

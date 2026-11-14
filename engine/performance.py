"""Computational performance tests (Chapter 3, Research Question 7; Chapter 4, Table 3).

    time and space   runtime (time.perf_counter) and peak memory (tracemalloc) of each stage:
                     augmented IRT, augmented BKT, Random Forest training and SHAP
    live scoring     time to score ONE answer (IRT ability update + BKT mastery update)
    robustness       Random Forest F1 on clean data vs data with 10% of answers flipped (noise)
                     or 10% of answers removed (missing); criterion: drop <= 5 percentage points
    scalability      runtime and memory as the data grow from 50 to 500 users;
                     criterion: runtime grows no faster than roughly linearly (log-log slope <= 1.2)
"""
from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pandas as pd

from . import bkt as bkt_mod
from . import irt as irt_mod
from . import predict as pred_mod
from .evaluate import response_matrices
from .simulate import simulate_study

F1_DROP_LIMIT_PP = 5.0
SLOPE_LIMIT = 1.2
SCORE_LIMIT_MS = 1000.0


def _measure(fn, *args, **kw):
    """Run fn once for time (no tracing) and once under tracemalloc for peak memory."""
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    seconds = time.perf_counter() - t0
    tracemalloc.start()
    fn(*args, **kw)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, seconds, peak / 2 ** 20


def _fit_irt(responses):
    X, C = response_matrices(responses)
    m = irt_mod.AugmentedIRT("2pl", augmented=True)
    m.fit(X.to_numpy(float), C.to_numpy(float))
    return m, pd.Series(m.result.theta, index=X.index), list(X.columns)


def _fit_bkt(responses):
    return bkt_mod.BKT(augmented=True).fit(responses)


def _rf_and_shap(Xf, y, seed=0):
    return pred_mod.shap_ranking(Xf, y, seed=seed)


# ---------------------------------------------------------------------- stages
def pipeline_profile(responses: pd.DataFrame, users: pd.DataFrame, seed=0) -> dict:
    """Runtime (s) and peak memory (MB) of every stage of the algorithm on one dataset."""
    import shap  # noqa: F401  (load the library first so its import time is not counted)
    (irt_m, theta, items), t_irt, m_irt = _measure(_fit_irt, responses)
    bkt_m, t_bkt, m_bkt = _measure(_fit_bkt, responses)
    mastery = bkt_m.predict(responses)
    Xf, y = pred_mod.build_features(responses, users, theta, mastery)
    _, t_rf, m_rf = _measure(_rf_and_shap, Xf, y, seed)
    return {"n_users": int(responses["user_id"].nunique()), "n_responses": int(len(responses)),
            "irt_seconds": t_irt, "irt_peak_mb": m_irt, "bkt_seconds": t_bkt, "bkt_peak_mb": m_bkt,
            "rf_shap_seconds": t_rf, "rf_shap_peak_mb": m_rf,
            "total_seconds": t_irt + t_bkt + t_rf, "peak_mb": max(m_irt, m_bkt, m_rf),
            "_models": (irt_m, bkt_m, items)}


def live_scoring_ms(irt_m, bkt_m, items, n_calls=500, seed=0) -> dict:
    """Average time to score one answer, as the app does after every comprehension check."""
    rng = np.random.default_rng(seed)
    k = min(12, len(items))
    resp = {j: int(rng.random() < 0.6) for j in range(k)}
    conf = {j: int(rng.integers(1, 6)) for j in range(k)}
    skill = next(iter(bkt_m.params))
    t0 = time.perf_counter()
    for _ in range(n_calls):
        irt_m.score_user(resp, conf)
    irt_ms = (time.perf_counter() - t0) / n_calls * 1000
    t0 = time.perf_counter()
    L = bkt_m.params[skill].L0
    for i in range(n_calls):
        L = bkt_m.update(L, i % 2, i % 3, skill)
    bkt_ms = (time.perf_counter() - t0) / n_calls * 1000
    return {"irt_ms": irt_ms, "bkt_ms": bkt_ms, "per_answer_ms": irt_ms + bkt_ms,
            "passed": irt_ms + bkt_ms < SCORE_LIMIT_MS}


# ---------------------------------------------------------------------- robustness
def perturb(responses: pd.DataFrame, kind: str, rate=0.10, seed=0) -> pd.DataFrame:
    """kind='noise': flip `rate` of the answers; kind='missing': remove `rate` of them."""
    rng = np.random.default_rng(seed)
    out = responses.copy()
    pick = rng.random(len(out)) < rate
    if kind == "noise":
        out.loc[pick, "correct"] = 1 - out.loc[pick, "correct"]
    elif kind == "missing":
        out = out.loc[~pick]
    else:
        raise ValueError("kind must be 'noise' or 'missing'")
    return out.reset_index(drop=True)


def rf_f1(responses: pd.DataFrame, users: pd.DataFrame, n_repeats=3, seed=0) -> float:
    """Whole pipeline (IRT -> BKT -> features) then cross-validated Random Forest F1."""
    _, theta, _ = _fit_irt(responses)
    mastery = _fit_bkt(responses).predict(responses)
    Xf, y = pred_mod.build_features(responses, users, theta, mastery)
    cv = pred_mod.cross_validate(Xf, y, n_splits=5, n_repeats=n_repeats, seed=seed)
    return float(cv.loc[cv["model"] == "random_forest", "f1"].mean())


def robustness(responses, users, rate=0.10, n_repeats=3, seed=0) -> dict:
    clean = rf_f1(responses, users, n_repeats, seed)
    out = {"clean_f1": clean}
    for kind in ("noise", "missing"):
        f1 = rf_f1(perturb(responses, kind, rate, seed), users, n_repeats, seed)
        drop_pp = (clean - f1) * 100
        out[kind] = {"f1": f1, "drop_pp": drop_pp, "passed": drop_pp <= F1_DROP_LIMIT_PP}
    out["passed"] = out["noise"]["passed"] and out["missing"]["passed"]
    return out


# ---------------------------------------------------------------------- scalability
def scalability(sizes=(50, 100, 200, 300, 500), seed=0) -> dict:
    rows = []
    for n in sizes:
        responses, users = simulate_study(n_users=n, seed=seed + n)
        prof = pipeline_profile(responses, users, seed)
        irt_m, bkt_m, items = prof.pop("_models")
        prof.update({f"live_{k}": v for k, v in live_scoring_ms(irt_m, bkt_m, items, n_calls=200).items()
                     if k != "passed"})
        rows.append(prof)
    tab = pd.DataFrame(rows)
    x = np.log(tab["n_responses"])
    slopes = {c: float(np.polyfit(x, np.log(tab[c]), 1)[0])
              for c in ("irt_seconds", "bkt_seconds", "total_seconds")}
    return {"table": tab, "slopes": slopes,
            "passed": all(s <= SLOPE_LIMIT for s in slopes.values())}


def run(responses=None, users=None, sizes=(50, 100, 200, 300, 500), n_repeats=3, seed=0) -> dict:
    """All performance tests. Without data, a 150-user synthetic study is used."""
    if responses is None:
        responses, users = simulate_study(n_users=150, seed=7)
    prof = pipeline_profile(responses, users, seed)
    irt_m, bkt_m, items = prof.pop("_models")
    return {"pipeline": prof,
            "live_scoring": live_scoring_ms(irt_m, bkt_m, items),
            "robustness": robustness(responses, users, n_repeats=n_repeats, seed=seed),
            "scalability": scalability(sizes, seed)}


def report_markdown(r: dict) -> str:
    p, ls, rb, sc = r["pipeline"], r["live_scoring"], r["robustness"], r["scalability"]
    ok = lambda b: "PASS" if b else "FAIL"
    lines = ["# Table 3 – Computational performance", "",
             f"Dataset: {p['n_users']} users, {p['n_responses']} responses.", "",
             "## Time and space efficiency", "",
             "| Stage | Runtime (s) | Peak memory (MB) |", "|---|---|---|",
             f"| Augmented IRT | {p['irt_seconds']:.2f} | {p['irt_peak_mb']:.1f} |",
             f"| Augmented BKT | {p['bkt_seconds']:.2f} | {p['bkt_peak_mb']:.1f} |",
             f"| Random Forest + SHAP | {p['rf_shap_seconds']:.2f} | {p['rf_shap_peak_mb']:.1f} |",
             f"| **Full pipeline** | **{p['total_seconds']:.2f}** | **{p['peak_mb']:.1f}** |", "",
             f"Live scoring of one answer: IRT {ls['irt_ms']:.2f} ms + BKT {ls['bkt_ms']:.3f} ms = "
             f"**{ls['per_answer_ms']:.2f} ms** (limit {SCORE_LIMIT_MS:.0f} ms) → **{ok(ls['passed'])}**", "",
             "## Robustness (10% of answers changed)", "",
             "| Data | Random Forest F1 | Drop (percentage points) | Result |", "|---|---|---|---|",
             f"| Clean | {rb['clean_f1']:.3f} | — | — |",
             f"| 10% noise (answers flipped) | {rb['noise']['f1']:.3f} | {rb['noise']['drop_pp']:.1f} | {ok(rb['noise']['passed'])} |",
             f"| 10% missing (answers removed) | {rb['missing']['f1']:.3f} | {rb['missing']['drop_pp']:.1f} | {ok(rb['missing']['passed'])} |",
             "", f"Criterion: F1 drops by no more than {F1_DROP_LIMIT_PP:.0f} percentage points.", "",
             "## Scalability", "",
             "| Users | Responses | IRT (s) | BKT (s) | RF + SHAP (s) | Total (s) | Peak memory (MB) | Live scoring (ms) |",
             "|---|---|---|---|---|---|---|---|"]
    for _, t in sc["table"].iterrows():
        lines.append(f"| {int(t['n_users'])} | {int(t['n_responses'])} | {t['irt_seconds']:.2f} | {t['bkt_seconds']:.2f} | "
                     f"{t['rf_shap_seconds']:.2f} | {t['total_seconds']:.2f} | {t['peak_mb']:.1f} | {t['live_per_answer_ms']:.2f} |")
    s = sc["slopes"]
    lines += ["", f"Growth (log-log slope of runtime vs responses; 1.0 = linear): IRT {s['irt_seconds']:.2f}, "
                  f"BKT {s['bkt_seconds']:.2f}, total {s['total_seconds']:.2f} → **{ok(sc['passed'])}** "
                  f"(limit {SLOPE_LIMIT})", ""]
    return "\n".join(lines)

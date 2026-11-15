"""Tests for the algorithm-evaluation tools: parameter recovery, performance tests and the
ASSISTments benchmark.  Run with:  pytest -q"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import assistments, performance, recovery  # noqa: E402
from engine.simulate import simulate_study             # noqa: E402


# ---------------------------------------------------------------- parameter recovery
def test_recovery_table_has_four_parameters_and_valid_coverage():
    _, table = recovery.run(n_datasets=2, n_users=120)
    assert list(table["parameter"]) == ["lambda", "T_low", "T_medium", "T_high"]
    assert table["coverage_pct"].between(0, 100).all()
    assert (table["rmse"] >= 0).all()


def test_recovery_estimates_are_close_to_truth():
    _, table = recovery.run(n_datasets=3, n_users=150)
    assert abs(table.loc[0, "mean_estimate"] - recovery.TRUE_LAMBDA) < 0.3
    assert np.allclose(table["mean_estimate"].iloc[1:], recovery.TRUE_T, atol=0.04)


# ---------------------------------------------------------------- performance
def test_perturb_noise_flips_about_ten_percent():
    r, _ = simulate_study(n_users=40, seed=3)
    noisy = performance.perturb(r, "noise", 0.10, seed=1)
    changed = (noisy["correct"] != r["correct"]).mean()
    assert len(noisy) == len(r) and 0.06 < changed < 0.14


def test_perturb_missing_removes_about_ten_percent():
    r, _ = simulate_study(n_users=40, seed=3)
    fewer = performance.perturb(r, "missing", 0.10, seed=1)
    assert 0.86 * len(r) < len(fewer) < 0.94 * len(r)


def test_pipeline_profile_reports_time_and_memory():
    r, u = simulate_study(n_users=60, seed=5)
    p = performance.pipeline_profile(r, u)
    irt_m, bkt_m, items = p.pop("_models")
    assert p["total_seconds"] > 0 and p["peak_mb"] > 0
    live = performance.live_scoring_ms(irt_m, bkt_m, items, n_calls=20)
    assert live["passed"] and live["per_answer_ms"] > 0


# ---------------------------------------------------------------- ASSISTments
def _fake_assistments(path, n_users=30, seed=0):
    rng = np.random.default_rng(seed)
    rows, oid = [], 0
    for u in range(n_users):
        for skill in ("Addition", "Fractions"):
            for k in range(6):
                oid += 1
                rows.append({"order_id": oid, "user_id": u, "problem_id": rng.integers(0, 12),
                             "skill_name": skill, "correct": int(rng.random() < 0.4 + 0.08 * k), "original": 1})
        rows.append({"order_id": oid + 10_000, "user_id": u, "problem_id": 99, "skill_name": "Addition",
                     "correct": 1, "original": 0})                     # scaffolding row: must be dropped
    pd.DataFrame(rows).to_csv(path, index=False)


def test_assistments_loader_and_benchmark(tmp_path):
    f = tmp_path / "as.csv"
    _fake_assistments(f)
    df = assistments.load(str(f))
    assert (df["problem_id"] != 99).all()
    r = assistments.benchmark(df, n_items=12, n_skills=2, n_repeats=1, fractions=(0.5, 1.0))
    assert r["irt"]["2pl"]["n_params"] == 2 * r["irt"]["n_items"]
    assert 0 <= r["bkt"]["standard"]["auc"] <= 1 and 0 <= r["bkt"]["forgetting"]["auc"] <= 1
    assert "Standard BKT" in assistments.report_markdown(r)

"""Unit tests for the evaluation engine.  Run with:  pytest -q"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import evaluate, predict                                  # noqa: E402
from engine.bkt import BKT                                            # noqa: E402
from engine.irt import AugmentedIRT, center_confidence, compare        # noqa: E402
from engine.simulate import simulate_bkt, simulate_irt, simulate_study  # noqa: E402
from engine.stats import corrected_resampled_ttest, kr20               # noqa: E402


# ---------------------------------------------------------------- IRT
def test_irt_gradient_matches_numerical():
    d = simulate_irt(n_users=40, n_items=8, seed=1)
    m = AugmentedIRT("2pl", True)
    mask = np.ones_like(d["X"], bool)
    M = center_confidence(d["conf"], mask)
    p = np.random.default_rng(0).normal(0, 0.3, 2 * 8 + 1)
    _, g = m._negloglik(p, d["X"], M, mask)
    num = np.array([(m._negloglik(p + e, d["X"], M, mask)[0] - m._negloglik(p - e, d["X"], M, mask)[0]) / 2e-5
                    for e in np.eye(len(p)) * 1e-5])
    assert np.allclose(g, num, atol=1e-4)


def test_lambda_is_recovered():
    d = simulate_irt(n_users=300, lam=1.0, seed=11)
    r = AugmentedIRT("2pl", True).fit(d["X"], d["conf"])
    assert abs(r.lam - 1.0) < 3 * r.lam_se


def test_standard_model_has_one_parameter_less():
    d = simulate_irt(n_users=80, seed=2)
    out = compare(d["X"], d["conf"])
    assert out["augmented"]["n_params"] - out["standard"]["n_params"] == 1
    assert out["augmented"]["loglik"] >= out["standard"]["loglik"] - 1e-6


def test_confidence_centered_within_user():
    conf = np.array([[5, 5, 5], [1, 3, 5]], float)
    m = center_confidence(conf, np.ones_like(conf, bool))
    assert np.allclose(m[0], 0) and np.isclose(m[1].sum(), 0)


def test_missing_answers_are_allowed():
    d = simulate_irt(n_users=60, seed=3)
    X = d["X"].copy(); X[::7, ::3] = np.nan
    r = AugmentedIRT("2pl", True).fit(X, d["conf"])
    assert np.isfinite(r.loglik)


def test_live_scoring_orders_users():
    d = simulate_irt(n_users=150, seed=4)
    m = AugmentedIRT("2pl", True); m.fit(d["X"], d["conf"])
    hi, _ = m.score_user({j: 1 for j in range(10)}, {j: 4 for j in range(10)})
    lo, _ = m.score_user({j: 0 for j in range(10)}, {j: 4 for j in range(10)})
    assert hi > lo


# ---------------------------------------------------------------- BKT
def test_bkt_recovers_learning_rates():
    df = simulate_bkt(n_users=300, steps_per_skill=12, T=(0.05, 0.12, 0.25), seed=21)
    m = BKT(True).fit(df)
    T = np.mean([m.params[s].T for s in m.params], axis=0)
    assert T[0] < T[1] < T[2]
    assert np.allclose(T, (0.05, 0.12, 0.25), atol=0.05)


def test_standard_bkt_has_equal_rates():
    df = simulate_bkt(n_users=50, seed=22)
    m = BKT(False).fit(df)
    for p in m.params.values():
        assert np.allclose(p.T, p.T[0])


def test_bkt_live_update_moves_mastery():
    df = simulate_bkt(n_users=100, seed=23)
    m = BKT(True).fit(df)
    L = m.params["SHAP"].L0
    assert m.update(L, 1, 2, "SHAP") > L                      # a correct answer raises mastery
    assert m.update(L, 1, 2, "SHAP") > m.update(L, 0, 2, "SHAP")  # correct beats incorrect


# ---------------------------------------------------------------- statistics
def test_corrected_ttest_is_more_conservative():
    rng = np.random.default_rng(0)
    a = rng.normal(0.70, 0.02, 50); b = a + rng.normal(0.01, 0.02, 50)
    corrected = corrected_resampled_ttest(a, b, n_train=48, n_test=12)
    from scipy import stats
    naive = stats.ttest_rel(b, a)
    assert corrected["p"] > naive.pvalue


def test_kr20_range():
    d = simulate_irt(n_users=200, seed=5)
    assert 0 < kr20(d["X"]) < 1


# ---------------------------------------------------------------- end to end
def test_full_evaluation_runs():
    responses, users = simulate_study(n_users=40, seed=3)
    r = evaluate.run(responses, users, n_repeats=1)
    for key in ("H1_irt", "H2_bkt", "H3_prediction", "H4_shap"):
        assert isinstance(r[key]["supported"], bool)
    assert "## H4" in evaluate.report_markdown(r)


def test_features_use_only_pre_outcome_data():
    responses, users = simulate_study(n_users=30, seed=4)
    assert "post_score" not in predict.FEATURES


# ---------------------------------------------------------------- algorithm tests (v7)
def test_rasch_baseline_has_fewer_params_and_fits():
    from engine.irt import AugmentedIRT
    from engine.simulate import simulate_irt
    d = simulate_irt(n_users=120, n_items=12, seed=3)
    r1 = AugmentedIRT("1pl", augmented=False).fit(d["X"])
    r2 = AugmentedIRT("2pl", augmented=False).fit(d["X"])
    assert r1.n_params == 12 and r2.n_params == 24
    assert np.allclose(r1.a, 1.0)
    assert r2.loglik >= r1.loglik - 1e-6          # 2PL nests 1PL


def test_rasch_augmented_lambda_se():
    from engine.irt import AugmentedIRT
    from engine.simulate import simulate_irt
    d = simulate_irt(n_users=150, n_items=12, lam=1.0, seed=4)
    r = AugmentedIRT("1pl", augmented=True).fit(d["X"], d["conf"])
    assert np.isfinite(r.lam_se) and r.lam_se > 0


def test_bkt_forgetting_recovers_small_F():
    from engine.bkt import BKT
    from engine.simulate import simulate_bkt
    df = simulate_bkt(n_users=200, steps_per_skill=10, seed=5)
    m = BKT(augmented=False, forgetting=True).fit(df)
    assert all(p.F < 0.1 for p in m.params.values())   # data were generated without forgetting


def test_bkt_lr_test_detects_trust_effect():
    from engine.bkt import likelihood_ratio_test
    from engine.simulate import simulate_bkt
    strong = simulate_bkt(n_users=300, steps_per_skill=10, T=(0.02, 0.15, 0.35), seed=6)
    none = simulate_bkt(n_users=300, steps_per_skill=10, T=(0.12, 0.12, 0.12), seed=6)
    assert likelihood_ratio_test(strong)["p"] < 0.05
    assert likelihood_ratio_test(none)["p"] > 0.01


def test_ablation_returns_four_configurations():
    from engine import evaluate
    from engine.simulate import simulate_study
    r, u = simulate_study(n_users=60, seed=2)
    X, C = evaluate.response_matrices(r)
    from engine.irt import AugmentedIRT
    import pandas as pd
    ts = pd.Series(AugmentedIRT(augmented=False).fit(X.to_numpy(float)).theta, index=X.index)
    ta = pd.Series(AugmentedIRT(augmented=True).fit(X.to_numpy(float), C.to_numpy(float)).theta, index=X.index)
    out = evaluate.ablation(r, u, ts, ta, n_splits=3, n_repeats=1)
    assert [row["config"] for row in out["table"]] == ["none", "lambda_only", "trust_only", "full"]
    assert set(out["tests"]) == {"none", "lambda_only", "trust_only"}

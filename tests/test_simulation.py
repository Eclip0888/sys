"""The simulation study runs and the methods behave as designed (small, fast settings)."""
import numpy as np

from sim import simulation as S


def test_generate_shapes_and_rates():
    pool = S.synthetic_pool(100, seed=1)
    Z, _, _ = S.standardize(pool)
    d = S.generate(Z, pool["qwords"].to_numpy(float), 300, 15, "high", np.random.default_rng(1))
    assert d.X.shape == d.U.shape == (300, 15)
    assert 0.15 < d.rapid.mean() < 0.35                        # Beta(2, 6) has mean 0.25
    assert (d.t[d.rapid] < 2).mean() > 0.7                     # rapid guesses are fast


def test_calibration_recovers_lltm_weights():
    pool = S.synthetic_pool(200, seed=2)
    Z, _, _ = S.standardize(pool)
    d = S.generate(Z, pool["qwords"].to_numpy(float), 1500, 15, "low", np.random.default_rng(12))
    est = S.calibrate(d, Z, True, True)
    assert np.allclose(est["beta"], S.TRUE["beta"], atol=0.15)


def test_full_algorithm_beats_raw_correctness():
    rep = S.run_study(S.synthetic_pool(), S.Config(replications=4, users=150, calibration_sets=1))
    m = rep["mean"]
    auc = lambda c, meth: m[(m.condition == c) & (m.method == meth)].auc.iloc[0]
    for cond in ("low", "high"):
        assert auc(cond, "full") > auc(cond, "raw")
    assert set(rep["decisions"]) == {"H1", "H2", "H3", "H4"}
    assert "Table 5" in S.report_markdown(rep)

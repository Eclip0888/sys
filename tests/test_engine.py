"""Correctness tests for the evaluation algorithm (Table 8, Correctness)."""
import math

import numpy as np
import pytest

from app import engine
from app.complexity import answer_features, plain_text, question_length

P = engine.Params()
MID = {"fkgl": P.mean[0], "len": P.mean[1], "tech": P.mean[2], "level": P.mean[3]}


def test_difficulty_at_feature_means_is_beta0():
    assert engine.difficulty(MID, P) == pytest.approx(P.beta0)


def test_difficulty_increases_with_each_feature():
    for k, step in zip(engine.FEATURE_ORDER, P.sd):
        harder = dict(MID, **{k: MID[k] + step})
        assert engine.difficulty(harder, P) > engine.difficulty(MID, P)


def test_difficulty_matches_lltm_formula():
    f = {"fkgl": 14.0, "len": 6.0, "tech": 0.3, "level": 3}
    z = [(f[k] - m) / s for k, m, s in zip(engine.FEATURE_ORDER, P.mean, P.sd)]
    assert engine.difficulty(f, P) == pytest.approx(P.beta0 + sum(b * x for b, x in zip(P.beta, z)))


def test_rapid_guess_threshold_and_weight():
    assert engine.rapid_guess_threshold(40) == pytest.approx(2.0)          # 0.10 * 13 = 1.3 -> floor of 2 s
    assert engine.rapid_guess_threshold(400) == pytest.approx(0.10 * 103)
    assert engine.response_weight(1.5, 40, P) == P.omega_min
    assert engine.response_weight(2.0, 40, P) == 1.0


def test_probability_of_correct_tap_has_guessing_floor_and_slip():
    assert engine.p_correct(-10, 0, P) == pytest.approx(P.c, abs=1e-4)
    assert engine.p_correct(10, 0, P) == pytest.approx(1 - P.s, abs=1e-4)


def test_posterior_is_prior_without_responses_and_mean_zero():
    post = engine.posterior([], P)
    assert post.sum() == pytest.approx(1.0)
    assert engine.eap(post)[0] == pytest.approx(0.0, abs=1e-9)


def test_eap_moves_up_after_correct_and_down_after_wrong():
    up = engine.eap(engine.posterior([{"x": 1, "b": 0.0, "w": 1.0}], P))[0]
    down = engine.eap(engine.posterior([{"x": 0, "b": 0.0, "w": 1.0}], P))[0]
    assert up > 0 > down


def test_small_weight_changes_ability_less():
    full = engine.eap(engine.posterior([{"x": 1, "b": 0.0, "w": 1.0}], P))[0]
    weak = engine.eap(engine.posterior([{"x": 1, "b": 0.0, "w": P.omega_min}], P))[0]
    assert 0 < weak < full


def test_weight_zero_means_no_evidence():
    post0 = engine.posterior([], P)
    c = engine.comprehension(post0, 0.0, True, 0.0, P)
    prior = float((engine.p_understand(engine.NODES, 0.0, P.a) * post0).sum())
    assert c == pytest.approx(prior)


def test_comprehension_bayes_rule_single_node():
    post = np.zeros_like(engine.NODES); post[20] = 1.0                    # theta = 0
    pi = 1 / (1 + math.exp(0))                                             # b = 0 -> 0.5
    expected = pi * (1 - P.s) / (pi * (1 - P.s) + (1 - pi) * P.c)
    assert engine.comprehension(post, 0.0, True, 1.0, P) == pytest.approx(expected)


def test_labels_and_thresholds():
    assert engine.label(0.70) == "Understood"
    assert engine.label(0.69) == "Partly understood"
    assert engine.label(0.40) == "Partly understood"
    assert engine.label(0.39) == "Not understood"


def test_evaluate_orders_outcomes_sensibly():
    careful_right = engine.evaluate([], MID, True, 12.0, 40, P)
    fast_right = engine.evaluate([], MID, True, 1.0, 40, P)
    wrong = engine.evaluate([], MID, False, 12.0, 40, P)
    assert careful_right.score > fast_right.score > wrong.score
    assert careful_right.label == "Understood" and wrong.label == "Not understood"
    assert fast_right.rapid and "guess" in fast_right.reason


def test_history_raises_confidence_for_hard_answers():
    hard = dict(MID, fkgl=MID["fkgl"] + 2 * P.sd[0], level=3)
    new_user = engine.evaluate([], hard, True, 12.0, 40, P).score
    strong_user = engine.evaluate([{"x": 1, "b": 0.5, "w": 1.0}] * 6, hard, True, 12.0, 40, P).score
    assert strong_user > new_user


def test_params_roundtrip():
    q = engine.Params.from_dict(P.to_dict())
    assert q == P


def test_complexity_features_and_markdown_stripping():
    md = "## Title\n\n- **First** point is here.\n- Second `code` point.\n\n```python\nx = 1\n```"
    assert "```" not in plain_text(md) and "x = 1" not in plain_text(md)
    easy = answer_features("The cat sat on the mat. The dog ran to the park. It was fun.")
    hard = answer_features("Photosynthetic organisms utilize chlorophyll-mediated electromagnetic absorption "
                           "to facilitate thermodynamically unfavorable carbohydrate biosynthesis.")
    assert hard["fkgl"] > easy["fkgl"] and hard["tech"] > easy["tech"]
    assert question_length("What is it?", ["A one", "B", "C", "D"]) == 8


def test_rapid_guess_never_labelled_understood():
    assert engine.label(0.95, rapid=True) == "Partly understood"
    easy = dict(MID, fkgl=MID["fkgl"] - 2 * P.sd[0], level=1)
    ev = engine.evaluate([{"x": 1, "b": 0.0, "w": 1.0}] * 5, easy, False, 0.5, 40, P)
    assert ev.rapid and ev.label != "Understood"

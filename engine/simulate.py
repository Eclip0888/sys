"""Synthetic data with KNOWN parameter values, used to check that the engine recovers them.

The same generator produces a complete study dataset in the format the app stores:
one row per answered comprehension question.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

SKILLS = ("SHAP", "LIME", "CF")
TRUST_LEVELS = ("low", "medium", "high")


def trust_level(rating: int) -> str:
    """1-2 = low, 3 = medium, 4-5 = high."""
    return "low" if rating <= 2 else ("medium" if rating == 3 else "high")


# ---------------------------------------------------------------------- IRT part
def simulate_irt(n_users=150, n_items=24, lam=0.8, model="2pl", n_options=4, seed=0):
    """Responses from the augmented IRT model with a known metacognitive weight lam.

    Confidence = general self-confidence (related to ability) + item-level feeling of
    knowing.  Only the within-user part enters the response model through lam.
    """
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 1, n_users)
    a = rng.lognormal(0.0, 0.3, n_items)
    b = rng.normal(0, 1, n_items)
    c = 1 / n_options if model == "3pl" else 0.0
    general = 0.6 * theta[:, None] + rng.normal(0, 0.5, (n_users, 1))
    conf = np.clip(np.rint(3 + general + rng.normal(0, 1.0, (n_users, n_items))), 1, 5)
    m = (conf - conf.mean(1, keepdims=True)) / 4
    P = c + (1 - c) * expit(a * (theta[:, None] - b) + lam * m)
    X = (rng.random((n_users, n_items)) < P).astype(float)
    return {"X": X, "conf": conf, "theta": theta, "a": a, "b": b, "lam": lam}


# ---------------------------------------------------------------------- BKT part
def simulate_bkt(n_users=150, steps_per_skill=10, L0=0.3, T=(0.05, 0.12, 0.20), G=0.2, S=0.1,
                 trust_bias_sd=1.0, seed=0):
    """Learning sequences where the learning rate depends on the trust level of each step.

    Returns a long DataFrame: user_id, skill, order, correct, trust (1-5), trust_level.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        tb = rng.normal(0, trust_bias_sd)
        for skill in SKILLS:
            learned = rng.random() < L0
            for t in range(steps_per_skill):
                rating = int(np.clip(np.rint(3 + tb + rng.normal(0, 1)), 1, 5))
                lvl = trust_level(rating)
                p = (1 - S) if learned else G
                correct = int(rng.random() < p)
                rows.append((f"U{u:03d}", skill, t, correct, rating, lvl))
                if not learned and rng.random() < T[TRUST_LEVELS.index(lvl)]:
                    learned = True
    return pd.DataFrame(rows, columns=["user_id", "skill", "order", "correct", "trust", "trust_level"])


# ---------------------------------------------------------------------- full study dataset
def simulate_study(n_users=60, n_items=24, sessions=3, lam=0.8, T=(0.05, 0.12, 0.20), seed=1):
    """A complete synthetic dataset shaped like the app's export.

    responses: user_id, session, order, item_id, skill, complexity, correct, confidence, trust
    users:     user_id, prior_knowledge (0-10), year_level, ai_courses, post_score (0-1)
    """
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 1, n_users)
    a = rng.lognormal(0, 0.3, n_items)
    b = rng.normal(0, 1, n_items)
    skills = [SKILLS[j % 3] for j in range(n_items)]
    complexity = ["L" if (j // 3) % 2 == 0 else "H" for j in range(n_items)]
    b = b + np.where(np.array(complexity) == "H", 0.5, -0.3)
    prior = np.clip(np.rint(5 + 1.5 * theta + rng.normal(0, 1.5, n_users)), 0, 10)
    trust_bias = rng.normal(0, 0.8, n_users)
    learned = {(u, s): rng.random() < 0.3 for u in range(n_users) for s in SKILLS}
    rows = []
    per_session = n_items // sessions
    for u in range(n_users):
        general = 0.6 * theta[u] + rng.normal(0, 0.5)
        order = 0
        for sess in range(sessions):
            for j in range(sess * per_session, (sess + 1) * per_session):
                s = skills[j]
                conf = int(np.clip(np.rint(3 + general + rng.normal(0, 1)), 1, 5))
                trust = int(np.clip(np.rint(3 + trust_bias[u] + rng.normal(0, 1)), 1, 5))
                m = (conf - 3 - general) / 4
                base = 1.6 if learned[(u, s)] else -1.0          # knowledge state (BKT part)
                p = expit(base + 0.6 * a[j] * (theta[u] - b[j]) + lam * m)
                correct = int(rng.random() < p)
                rows.append((f"U{u:03d}", sess + 1, order, f"Q{j + 1:02d}", s, complexity[j], correct, conf, trust))
                order += 1
                if not learned[(u, s)] and rng.random() < T[TRUST_LEVELS.index(trust_level(trust))]:
                    learned[(u, s)] = True
    responses = pd.DataFrame(rows, columns=["user_id", "session", "order", "item_id", "skill", "complexity",
                                            "correct", "confidence", "trust"])
    mastery = np.array([np.mean([learned[(u, s)] for s in SKILLS]) for u in range(n_users)])
    post = np.clip(expit(0.9 * theta + 1.2 * mastery + 0.15 * (prior - 5) - 0.2 + rng.normal(0, 0.35, n_users)), 0, 1)
    users = pd.DataFrame({"user_id": [f"U{u:03d}" for u in range(n_users)], "prior_knowledge": prior,
                          "year_level": rng.integers(2, 5, n_users), "ai_courses": rng.integers(0, 3, n_users),
                          "post_score": post})
    return responses, users

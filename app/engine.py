"""Real-time comprehension evaluation: the augmented IRT algorithm (Chapter 3, Steps 5-8).

1. Difficulty from answer complexity (LLTM):   b = beta0 + sum_k beta_k * z_k
2. Probability of understanding:               P(U = 1 | theta, b) = 1 / (1 + exp(-a (theta - b)))
   Probability of a correct tap (4 options):   P(X = 1 | theta, b) = c + (1 - c - s) * P(U = 1 | theta, b)
                                               c = 0.25 (chance with 4 options), s = 0.05 (slip)
3. Response-time weight (rapid guessing):      w = omega_min if t < t_g else 1,  t_g = max(2, 0.10 * (3 + L/4))
4. Weighted EAP ability over quadrature nodes: p(theta | data) ∝ phi(theta) * prod_j P(X_j | theta, b_j)^w_j
5. Comprehension score: the probability that the user understood THIS answer, given the tap and the
   earlier responses (Bayes' rule over the hidden understanding U and the ability nodes):
       C = sum_k p_k pi_k L1^w / sum_k p_k [pi_k L1^w + (1 - pi_k) L0^w]
   p_k = posterior from earlier responses, pi_k = P(U = 1 | theta_k, b),
   L1 = P(X | U = 1) = 1 - s if correct else s,  L0 = P(X | U = 0) = c if correct else 1 - c
   Label: Understood >= 0.70 > Partly understood >= 0.40 > Not understood; a likely rapid guess is at most
   "Partly understood", because a guess cannot confirm understanding
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

NODES = np.linspace(-4, 4, 41)
PRIOR = np.exp(-0.5 * NODES ** 2)
PRIOR /= PRIOR.sum()
FEATURE_ORDER = ("fkgl", "len", "tech", "level")


@dataclass
class Params:
    """Calibrated parameters.  Defaults are the simulation's true values until a calibration replaces them."""
    beta0: float = 0.0
    beta: tuple = (0.5, 0.3, 0.4, 0.5)          # FKGL, LEN, TECH, LEVEL (standardized features)
    a: float = 1.2                              # common discrimination
    omega_min: float = 0.2                      # weight of a likely rapid guess
    c: float = 0.25                             # chance of a correct tap without understanding (4 options)
    s: float = 0.05                             # slip: wrong tap despite understanding
    mean: tuple = (11.0, 5.4, 0.22, 2.0)        # feature means for standardization
    sd: tuple = (3.0, 0.6, 0.08, 0.8)           # feature standard deviations
    source: str = "default (simulation true values)"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = {k: (tuple(v) if isinstance(v, list) else v) for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        return cls(**d)


def standardize(features: dict, p: Params) -> np.ndarray:
    x = np.array([float(features.get(k, m)) for k, m in zip(FEATURE_ORDER, p.mean)])
    return (x - np.array(p.mean)) / np.array(p.sd)


def difficulty(features: dict, p: Params) -> float:
    """LLTM: difficulty of a new AI answer from its standardized complexity features."""
    return float(p.beta0 + np.dot(p.beta, standardize(features, p)))


def expected_time(question_words: int) -> float:
    return 3.0 + question_words / 4.0


def rapid_guess_threshold(question_words: int) -> float:
    return max(2.0, 0.10 * expected_time(question_words))


def response_weight(seconds: float, question_words: int, p: Params) -> float:
    return p.omega_min if seconds < rapid_guess_threshold(question_words) else 1.0


def p_understand(theta, b, a):
    return 1.0 / (1.0 + np.exp(-a * (np.asarray(theta) - b)))


def p_correct(theta, b, p: "Params"):
    return p.c + (1.0 - p.c - p.s) * p_understand(theta, b, p.a)


def posterior(responses: list[dict], p: Params) -> np.ndarray:
    """Weighted EAP posterior over NODES. Each response: {'x': 0/1, 'b': float, 'w': float}."""
    logp = np.log(PRIOR)
    for r in responses:
        pr = np.clip(p_correct(NODES, r["b"], p), 1e-9, 1 - 1e-9)
        logp = logp + r["w"] * (np.log(pr) if r["x"] else np.log(1 - pr))
    post = np.exp(logp - logp.max())
    return post / post.sum()


def eap(post: np.ndarray) -> tuple[float, float]:
    mean = float((NODES * post).sum())
    return mean, float(np.sqrt(((NODES - mean) ** 2 * post).sum()))


def comprehension(post_before: np.ndarray, b: float, correct: bool, w: float, p: Params) -> float:
    """Probability that the user understood THIS answer, given the tap (weighted by w) and earlier responses."""
    pi = p_understand(NODES, b, p.a)
    l1 = (1 - p.s) if correct else p.s
    l0 = p.c if correct else (1 - p.c)
    num = (post_before * pi * l1 ** w).sum()
    den = num + (post_before * (1 - pi) * l0 ** w).sum()
    return float(num / den)


def label(c: float, rapid: bool = False) -> str:
    """Understood >= 0.70 > Partly understood >= 0.40 > Not understood.  A likely rapid guess cannot confirm
    understanding, so its label is at most "Partly understood" (the score itself is unchanged)."""
    if c >= 0.70 and not rapid:
        return "Understood"
    return "Partly understood" if c >= 0.40 else "Not understood"


def difficulty_level(b: float) -> str:
    return "low" if b < -0.5 else ("high" if b > 0.5 else "moderate")


def reason(b: float, correct: bool, rapid: bool, theta: float, score: float = 1.0) -> str:
    level = difficulty_level(b)
    parts = [f"The AI answer had {level} complexity."]
    if correct and rapid:
        parts.append("Your answer was correct but very fast, so it may have been a guess and counted less. "
                     "Answer at your normal reading pace to confirm your understanding.")
    elif correct and score < 0.70:
        parts.append("You answered correctly, but with four options one correct tap can also be a lucky guess, "
                     "so a few more correct answers on complex topics will confirm your understanding.")
    elif correct:
        parts.append("You answered the question about it correctly at a normal reading pace.")
    elif rapid:
        parts.append("Your answer was incorrect and very fast, so it may have been a guess and counted less. "
                     "Re-read the AI answer, then try the next question at your normal reading pace.")
    else:
        parts.append("You answered the question about it incorrectly.")
    if theta > 0.5:
        parts.append("Your earlier answers also show good understanding of AI answers.")
    elif theta < -0.5:
        parts.append("Your earlier answers suggest you may want to read AI answers more slowly.")
    return " ".join(parts)


@dataclass
class Evaluation:
    b: float
    w: float
    rapid: bool
    theta: float
    theta_se: float
    score: float
    label: str
    reason: str
    threshold: float = field(default=0.0)


def evaluate(history: list[dict], features: dict, correct: bool, seconds: float, question_words: int,
             p: Params) -> Evaluation:
    """Score one new response given the user's earlier responses (each {'x','b','w'})."""
    b = difficulty(features, p)
    w = response_weight(seconds, question_words, p)
    c = comprehension(posterior(history, p), b, correct, w, p)
    post = posterior(history + [{"x": int(correct), "b": b, "w": w}], p)
    theta, se = eap(post)
    rapid = w < 1.0
    return Evaluation(b=b, w=w, rapid=rapid, theta=theta, theta_se=se, score=c, label=label(c, rapid),
                      reason=reason(b, correct, rapid, theta, c), threshold=rapid_guess_threshold(question_words))

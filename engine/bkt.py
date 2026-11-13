"""Bayesian Knowledge Tracing (BKT), standard and trust-conditioned.

Each skill (SHAP, LIME, CF) is a two-state hidden Markov model (not learned / learned):

    P(L0)          initial probability of mastery
    P(G), P(S)     guess and slip
    P(T)           learning rate                       (standard BKT)
    P(T | tau)     one learning rate per trust level   (augmented BKT: low, medium, high)

After the response at step t, the user moves from "not learned" to "learned" with the
learning rate for the trust level tau_t of the explanation shown at step t.  Standard BKT
is the special case where the three rates are equal.

Baseline with forgetting (forgetting=True): a learned user can move back to "not learned"
with probability P(F) after each step (Khajah et al.'s BKT+forget).  It is used only as an
additional baseline for the augmented model.

Parameters are estimated by Expectation-Maximization (Baum-Welch).  Guess and slip are
capped at 0.4 to avoid degenerate solutions where "learned" users are more often wrong.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LEVELS = ("low", "medium", "high")
CAP_GS = 0.4


def trust_level(rating: float) -> str:
    return "low" if rating <= 2 else ("medium" if rating == 3 else "high")


@dataclass
class SkillParams:
    L0: float
    T: np.ndarray     # length 3 (low, medium, high); equal values for standard BKT
    G: float
    S: float
    F: float = 0.0    # forgetting (0 unless forgetting=True)

    def as_dict(self):
        return {"L0": self.L0, "T_low": float(self.T[0]), "T_medium": float(self.T[1]),
                "T_high": float(self.T[2]), "G": self.G, "S": self.S, "F": self.F}


def _sequences(df: pd.DataFrame, skill: str):
    """List of (correct array, trust-level index array) per user for one skill."""
    sub = df[df["skill"] == skill].sort_values(["user_id", "order"])
    seqs = []
    for _, g in sub.groupby("user_id", sort=False):
        lv = g["trust_level"] if "trust_level" in g else g["trust"].map(trust_level)
        seqs.append((g["correct"].to_numpy(int), lv.map({k: i for i, k in enumerate(LEVELS)}).to_numpy(int)))
    return seqs


class BKT:
    def __init__(self, augmented: bool = True, forgetting: bool = False, max_iter: int = 200, tol: float = 1e-5):
        self.augmented, self.forgetting, self.max_iter, self.tol = augmented, forgetting, max_iter, tol
        self.params: dict[str, SkillParams] = {}
        self.n_eff: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ fitting (vectorized)
    @staticmethod
    def _pad(seqs):
        n = len(seqs); L = max(len(o) for o, _ in seqs)
        obs = np.zeros((n, L), int); lv = np.zeros((n, L), int); mask = np.zeros((n, L), bool)
        for i, (o, l) in enumerate(seqs):
            obs[i, :len(o)], lv[i, :len(o)], mask[i, :len(o)] = o, l, True
        return obs, lv, mask

    def _em_step(self, obs, lv, mask, p: SkillParams):
        n, L = obs.shape
        E0 = np.where(obs == 1, p.G, 1 - p.G); E1 = np.where(obs == 1, 1 - p.S, p.S)
        E0 = np.where(mask, E0, 1.0); E1 = np.where(mask, E1, 1.0)
        Tt = p.T[lv]                                   # rate applied after step t
        a0 = np.zeros((n, L)); a1 = np.zeros((n, L)); sc = np.ones((n, L))
        a0[:, 0] = (1 - p.L0) * E0[:, 0]; a1[:, 0] = p.L0 * E1[:, 0]
        sc[:, 0] = a0[:, 0] + a1[:, 0]; a0[:, 0] /= sc[:, 0]; a1[:, 0] /= sc[:, 0]
        for t in range(1, L):
            T = Tt[:, t - 1]
            q0 = a0[:, t - 1] * (1 - T) + a1[:, t - 1] * p.F
            q1 = a0[:, t - 1] * T + a1[:, t - 1] * (1 - p.F)
            x0, x1 = q0 * E0[:, t], q1 * E1[:, t]
            s_ = x0 + x1
            m = mask[:, t]
            sc[:, t] = np.where(m, s_, 1.0)
            a0[:, t] = np.where(m, x0 / s_, a0[:, t - 1]); a1[:, t] = np.where(m, x1 / s_, a1[:, t - 1])
        b0 = np.ones((n, L)); b1 = np.ones((n, L))
        for t in range(L - 2, -1, -1):
            T = Tt[:, t]; m = mask[:, t + 1]
            nb0 = ((1 - T) * E0[:, t + 1] * b0[:, t + 1] + T * E1[:, t + 1] * b1[:, t + 1]) / sc[:, t + 1]
            nb1 = (p.F * E0[:, t + 1] * b0[:, t + 1] + (1 - p.F) * E1[:, t + 1] * b1[:, t + 1]) / sc[:, t + 1]
            b0[:, t] = np.where(m, nb0, 1.0); b1[:, t] = np.where(m, nb1, 1.0)
        g0 = a0 * b0; g1 = a1 * b1; z = g0 + g1; g0 /= z; g1 /= z
        g0 = np.where(mask, g0, 0); g1 = np.where(mask, g1, 0)
        xi = np.zeros((n, L)); gd = np.zeros((n, L)); xf = np.zeros((n, L)); gf = np.zeros((n, L))
        if L > 1:
            pair = mask[:, 1:]
            xi[:, :-1] = np.where(pair, a0[:, :-1] * Tt[:, :-1] * E1[:, 1:] * b1[:, 1:] / sc[:, 1:], 0)
            gd[:, :-1] = np.where(pair, g0[:, :-1], 0)
            xf[:, :-1] = np.where(pair, a1[:, :-1] * p.F * E0[:, 1:] * b0[:, 1:] / sc[:, 1:], 0)
            gf[:, :-1] = np.where(pair, g1[:, :-1], 0)
        ll = np.log(np.where(mask, sc, 1.0)).sum()
        Tn = np.bincount(lv.ravel(), weights=xi.ravel(), minlength=3)
        Td = np.bincount(lv.ravel(), weights=gd.ravel(), minlength=3)
        stats_ = dict(L0=g1[:, 0].sum(), Gn=(g0 * obs).sum(), Gd=g0.sum(), Sn=(g1 * (1 - obs) * mask).sum(),
                      Sd=g1.sum(), Tn=Tn, Td=Td, Fn=xf.sum(), Fd=gf.sum())
        return stats_, ll

    def fit(self, df: pd.DataFrame):
        """df columns: user_id, skill, order, correct, trust (1-5) or trust_level."""
        self.params, self.n_eff = {}, {}
        for skill in sorted(df["skill"].unique()):
            obs, lv, mask = self._pad(_sequences(df, skill))
            p = SkillParams(0.3, np.full(3, 0.1), 0.2, 0.1, 0.05 if self.forgetting else 0.0)
            prev = -np.inf
            for _ in range(self.max_iter):
                st, ll = self._em_step(obs, lv, mask, p)
                Tn, Td = st["Tn"], st["Td"]
                if self.augmented:
                    T = np.where(Td > 0, Tn / np.maximum(Td, 1e-12), Tn.sum() / max(Td.sum(), 1e-12))
                else:
                    T = np.full(3, Tn.sum() / max(Td.sum(), 1e-12))
                p = SkillParams(float(np.clip(st["L0"] / len(obs), 1e-4, 1 - 1e-4)), np.clip(T, 1e-4, 1 - 1e-4),
                                float(np.clip(st["Gn"] / max(st["Gd"], 1e-12), 1e-4, CAP_GS)),
                                float(np.clip(st["Sn"] / max(st["Sd"], 1e-12), 1e-4, CAP_GS)),
                                float(np.clip(st["Fn"] / max(st["Fd"], 1e-12), 1e-4, CAP_GS)) if self.forgetting else 0.0)
                if abs(ll - prev) < self.tol:
                    break
                prev = ll
            p.loglik = ll
            self.params[skill] = p
            self.n_eff[skill] = Td if self.augmented else np.full(3, Td.sum())
        return self

    def t_standard_errors(self, skill):
        """Approximate SE of each learning rate (binomial, using the expected exposures)."""
        p, n = self.params[skill], self.n_eff[skill]
        return np.sqrt(p.T * (1 - p.T) / np.maximum(n, 1e-9))

    # ------------------------------------------------------------------ prediction
    def predict_sequence(self, obs, lv, skill):
        """P(correct) at each step given only earlier responses, and P(learned) after each step."""
        p = self.params[skill]
        L = p.L0; pc = np.zeros(len(obs)); mastery = np.zeros(len(obs))
        for t, (o, l) in enumerate(zip(obs, lv)):
            pc[t] = L * (1 - p.S) + (1 - L) * p.G
            L = self.update(L, o, l, skill)
            mastery[t] = L
        return pc, mastery

    def update(self, L, correct, level_idx, skill):
        """One live update: observe a response, then apply learning at this trust level."""
        p = self.params[skill]
        if correct:
            post = L * (1 - p.S) / (L * (1 - p.S) + (1 - L) * p.G)
        else:
            post = L * p.S / (L * p.S + (1 - L) * (1 - p.G))
        return post * (1 - p.F) + (1 - post) * p.T[level_idx]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = []
        for skill in sorted(df["skill"].unique()):
            if skill not in self.params:
                continue
            sub = df[df["skill"] == skill].sort_values(["user_id", "order"])
            for uid, g in sub.groupby("user_id", sort=False):
                lv = (g["trust_level"] if "trust_level" in g else g["trust"].map(trust_level))
                lv = lv.map({k: i for i, k in enumerate(LEVELS)}).to_numpy(int)
                pc, mastery = self.predict_sequence(g["correct"].to_numpy(int), lv, skill)
                out.append(pd.DataFrame({"user_id": uid, "skill": skill, "order": g["order"].to_numpy(),
                                         "correct": g["correct"].to_numpy(), "p_correct": pc, "mastery": mastery}))
        return pd.concat(out, ignore_index=True)


def metrics(pred: pd.DataFrame) -> dict:
    y, p = pred["correct"].to_numpy(), pred["p_correct"].to_numpy()
    return {"auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
            "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
            "accuracy": float(np.mean((p >= 0.5) == y))}


def likelihood_ratio_test(df: pd.DataFrame) -> dict:
    """Do the three trust-level learning rates differ?  Standard BKT (one rate per skill) is
    nested in augmented BKT (three rates per skill), so LR ~ chi2 with df = 2 x skills."""
    from scipy import stats as _st
    s_, a_ = BKT(augmented=False, tol=1e-7).fit(df), BKT(augmented=True, tol=1e-7).fit(df)
    ll_s = sum(p.loglik for p in s_.params.values()); ll_a = sum(p.loglik for p in a_.params.values())
    lr = max(2 * (ll_a - ll_s), 0.0); dof = 2 * len(a_.params)
    return {"loglik_standard": float(ll_s), "loglik_augmented": float(ll_a), "lr": float(lr), "df": dof,
            "p": float(_st.chi2.sf(lr, dof))}


VARIANTS = (("standard", False, False), ("forgetting", False, True), ("augmented", True, False))


def cross_validate(df: pd.DataFrame, n_splits=5, n_repeats=10, seed=0, variants=VARIANTS) -> pd.DataFrame:
    """Standard BKT, BKT with forgetting (baseline) and augmented BKT, with folds split by user."""
    users = np.array(sorted(df["user_id"].unique()))
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(n_repeats):
        perm = rng.permutation(users)
        folds = np.array_split(perm, n_splits)
        for k, test_users in enumerate(folds):
            train = df[~df["user_id"].isin(test_users)]
            test = df[df["user_id"].isin(test_users)]
            for name, aug, fg in variants:
                m = BKT(augmented=aug, forgetting=fg).fit(train)
                r = metrics(m.predict(test))
                rows.append({"repeat": rep, "fold": k, "model": name, **r,
                             "n_train": len(users) - len(test_users), "n_test": len(test_users)})
    return pd.DataFrame(rows)

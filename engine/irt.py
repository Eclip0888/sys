"""Augmented Item Response Theory (IRT) with a metacognitive weight.

Model (one response of user i to item j):

    P(x_ij = 1) = c_j + (1 - c_j) * sigmoid( a_j * (theta_i - b_j) + lam * m_ij )

    theta_i  ability of user i, assumed N(0, 1) and integrated out (marginal likelihood)
    a_j      discrimination of item j (kept positive: a_j = exp(log_a_j))
    b_j      difficulty of item j
    c_j      pseudo-guessing, fixed at 1/k for "3pl" or 0 for "2pl"/"1pl" (not estimated)
             "1pl" is the Rasch baseline: every a_j is fixed at 1, only b_j is estimated.
    m_ij     the user's confidence on this item, centered WITHIN the user:
             (conf_ij - mean of user i's confidences) / (scale range).  Centering within
             the user removes general self-confidence, so lam measures metacognitive
             resolution: whether the user is more often right on the items they felt
             surer about than usual.
    lam      metacognitive weight. lam = 0 gives the standard model, so the standard and
             augmented models are nested and differ by exactly one parameter.

Estimation: marginal maximum likelihood over a Gauss-Hermite grid for theta, with weak
priors on log a (N(0, 1)) and b (N(0, 2^2)) for stability on small samples, maximized
with L-BFGS-B using the analytic gradient.  Scoring of a single user (live use in the
app) is by expected a posteriori (EAP) estimation with the item parameters held fixed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats
from scipy.special import expit

N_QUAD = 41


def _quadrature(n: int = N_QUAD):
    """Nodes and weights for theta ~ N(0, 1)."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / w.sum()


def center_confidence(conf: np.ndarray, mask: np.ndarray, low: int = 1, high: int = 5) -> np.ndarray:
    """Within-user centered confidence, scaled to roughly [-1, 1]. Missing cells -> 0."""
    conf = np.where(mask, conf, np.nan).astype(float)
    user_mean = np.nanmean(np.where(mask, conf, np.nan), axis=1, keepdims=True)
    user_mean = np.where(np.isnan(user_mean), (low + high) / 2, user_mean)
    m = (conf - user_mean) / (high - low)
    return np.where(mask, m, 0.0)


@dataclass
class IRTResult:
    model: str                    # "2pl" or "3pl"
    augmented: bool
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    lam: float
    loglik: float                 # marginal log-likelihood (without priors)
    n_params: int
    n_persons: int
    theta: np.ndarray = field(default=None)       # EAP ability per person
    theta_se: np.ndarray = field(default=None)
    lam_se: float = float("nan")
    converged: bool = True

    @property
    def aic(self) -> float:
        return -2 * self.loglik + 2 * self.n_params

    @property
    def bic(self) -> float:
        return -2 * self.loglik + self.n_params * np.log(self.n_persons)


class AugmentedIRT:
    """Fit the standard (augmented=False) or augmented (augmented=True) IRT model."""

    def __init__(self, model: str = "2pl", augmented: bool = True, n_options: int = 4,
                 prior_sd_loga: float = 1.0, prior_sd_b: float = 2.0, n_quad: int = N_QUAD):
        if model not in ("1pl", "2pl", "3pl"):
            raise ValueError("model must be '1pl', '2pl' or '3pl'")
        self.rasch = model == "1pl"
        self.model, self.augmented = model, augmented
        self.c_value = 1.0 / n_options if model == "3pl" else 0.0
        self.prior_sd_loga, self.prior_sd_b = prior_sd_loga, prior_sd_b
        self.nodes, self.weights = _quadrature(n_quad)
        self.result: IRTResult | None = None

    # ------------------------------------------------------------------ core likelihood
    def _unpack(self, p, n_items):
        log_a, b = p[:n_items], p[n_items:2 * n_items]
        if self.rasch:
            log_a = np.zeros(n_items)
        lam = p[2 * n_items] if self.augmented else 0.0
        return np.exp(log_a), b, lam

    def _negloglik(self, p, X, M, mask, with_prior=True):
        n_items = X.shape[1]
        a, b, lam = self._unpack(p, n_items)
        c = self.c_value
        q = self.nodes[:, None, None]                                   # (Q,1,1)
        eta = a[None, None, :] * (q - b[None, None, :]) + lam * M[None]  # (Q,N,J)
        s = expit(eta)
        P = np.clip(c + (1 - c) * s, 1e-9, 1 - 1e-9)
        ll_qij = np.where(mask[None], X[None] * np.log(P) + (1 - X[None]) * np.log(1 - P), 0.0)
        ll_qi = ll_qij.sum(axis=2) + np.log(self.weights)[:, None]       # (Q,N)
        mx = ll_qi.max(axis=0)
        li = mx + np.log(np.exp(ll_qi - mx).sum(axis=0))                 # (N,)
        nll = -li.sum()
        # gradient
        r = np.exp(ll_qi - li[None])                                     # posterior (Q,N)
        g = np.where(mask[None], (X[None] - P) / (P * (1 - P)) * (1 - c) * s * (1 - s), 0.0)  # dll/deta
        rg = r[:, :, None] * g                                           # (Q,N,J)
        d_loga = -(rg * (q - b[None, None, :])).sum(axis=(0, 1)) * a
        d_b = (rg.sum(axis=(0, 1))) * a
        if self.rasch:
            d_loga = np.zeros_like(d_loga)
        grad = [d_loga, d_b]
        if self.augmented:
            grad.append(np.array([-(rg * M[None]).sum()]))
        grad = np.concatenate(grad)
        if with_prior:
            log_a = p[:n_items]
            if self.rasch:
                log_a = np.zeros_like(log_a)
            nll += 0.5 * np.sum(log_a ** 2) / self.prior_sd_loga ** 2 + 0.5 * np.sum(b ** 2) / self.prior_sd_b ** 2
            grad[:n_items] += log_a / self.prior_sd_loga ** 2
            grad[n_items:2 * n_items] += b / self.prior_sd_b ** 2
        return nll, grad

    # ------------------------------------------------------------------ fitting
    def fit(self, X, conf=None, mask=None, conf_low=1, conf_high=5, max_iter=1000, tol=1e-3):
        """X: (persons x items) 0/1 array (NaN = not answered). conf: same shape, 1..5."""
        X = np.asarray(X, dtype=float)
        if mask is None:
            mask = ~np.isnan(X)
        X = np.where(mask, X, 0.0)
        n, J = X.shape
        if self.augmented:
            if conf is None:
                raise ValueError("the augmented model needs confidence ratings")
            M = center_confidence(np.asarray(conf, dtype=float), mask, conf_low, conf_high)
        else:
            M = np.zeros_like(X)
        # starting values: b from proportion correct
        pc = np.clip((X * mask).sum(0) / np.maximum(mask.sum(0), 1), 0.05, 0.95)
        b0 = -np.log(pc / (1 - pc)) / 1.7
        p0 = np.concatenate([np.zeros(J), b0] + ([np.array([0.0])] if self.augmented else []))
        res = optimize.minimize(self._negloglik, p0, args=(X, M, mask, True), jac=True, method="L-BFGS-B",
                                options={"maxiter": max_iter, "ftol": tol * 1e-6, "gtol": 1e-5})
        a, b, lam = self._unpack(res.x, J)
        loglik = -self._negloglik(res.x, X, M, mask, with_prior=False)[0]
        k = (J if self.rasch else 2 * J) + (1 if self.augmented else 0)
        out = IRTResult(self.model, self.augmented, a, b, np.full(J, self.c_value), float(lam),
                        float(loglik), k, n, converged=bool(res.success))
        if self.augmented:
            out.lam_se = self._lam_se(res.x, X, M, mask)
        self._X, self._M, self._mask = X, M, mask
        out.theta, out.theta_se = self.eap(X, M, mask, a, b, lam)
        self.result = out
        return out

    def _lam_se(self, p, X, M, mask, h=1e-4):
        """Standard error of lam from the numerical Hessian of the penalized objective."""
        k = len(p)
        H = np.zeros((k, k))
        for i in range(k):
            e = np.zeros(k); e[i] = h
            H[i] = (self._negloglik(p + e, X, M, mask)[1] - self._negloglik(p - e, X, M, mask)[1]) / (2 * h)
        H = (H + H.T) / 2
        if self.rasch:                       # log a is fixed: drop those rows/columns
            J = X.shape[1]
            keep = np.r_[J:k]
            H = H[np.ix_(keep, keep)]
        try:
            cov = np.linalg.inv(H)
            return float(np.sqrt(max(cov[-1, -1], 0.0)))
        except np.linalg.LinAlgError:
            return float("nan")

    # ------------------------------------------------------------------ scoring
    def eap(self, X, M, mask, a, b, lam):
        q = self.nodes[:, None, None]
        P = np.clip(self.c_value + (1 - self.c_value) * expit(a * (q - b) + lam * M[None]), 1e-9, 1 - 1e-9)
        ll = np.where(mask[None], X[None] * np.log(P) + (1 - X[None]) * np.log(1 - P), 0.0).sum(2)
        ll += np.log(self.weights)[:, None]
        post = np.exp(ll - ll.max(0))
        post /= post.sum(0)
        mean = (post * self.nodes[:, None]).sum(0)
        sd = np.sqrt((post * (self.nodes[:, None] - mean) ** 2).sum(0))
        return mean, sd

    def score_user(self, responses: dict, confidences: dict | None = None, item_index: dict | None = None):
        """Live scoring for one user with fitted item parameters.

        responses: {item_id: 0/1}; confidences: {item_id: 1..5}; item_index: {item_id: column}.
        Returns (theta, se).
        """
        r = self.result
        if r is None:
            raise RuntimeError("fit the model (or load parameters) first")
        J = len(r.a)
        x = np.zeros((1, J)); mask = np.zeros((1, J), bool); conf = np.full((1, J), 3.0)
        for item, val in responses.items():
            j = item_index[item] if item_index else item
            x[0, j], mask[0, j] = val, True
            if confidences and item in confidences:
                conf[0, j] = confidences[item]
        M = center_confidence(conf, mask) if r.augmented else np.zeros_like(x)
        th, se = self.eap(x, M, mask, r.a, r.b, r.lam)
        return float(th[0]), float(se[0])

    # ------------------------------------------------------------------ diagnostics
    def predicted_prob(self):
        r = self.result
        return r.c + (1 - r.c) * expit(r.a * (r.theta[:, None] - r.b) + r.lam * self._M)

    def rmse(self):
        P = self.predicted_prob()
        return float(np.sqrt(np.sum(((self._X - P) ** 2) * self._mask) / self._mask.sum()))

    def person_fit_lz(self):
        """Standardized log-likelihood person-fit statistic lz (values < -1.96 = misfit)."""
        P = np.clip(self.predicted_prob(), 1e-9, 1 - 1e-9)
        X, mask = self._X, self._mask
        l0 = np.where(mask, X * np.log(P) + (1 - X) * np.log(1 - P), 0).sum(1)
        e = np.where(mask, P * np.log(P) + (1 - P) * np.log(1 - P), 0).sum(1)
        v = np.where(mask, P * (1 - P) * np.log(P / (1 - P)) ** 2, 0).sum(1)
        return (l0 - e) / np.sqrt(np.maximum(v, 1e-12))


def likelihood_ratio_test(standard: IRTResult, augmented: IRTResult) -> dict:
    """LR test of augmented vs standard (nested; df = difference in parameters)."""
    lr = max(2 * (augmented.loglik - standard.loglik), 0.0)
    df = augmented.n_params - standard.n_params
    return {"lr": lr, "df": df, "p": float(stats.chi2.sf(lr, df))}


def baseline_rasch(X):
    """Rasch (1PL) baseline: fit, AIC and BIC for the baseline comparison table."""
    m = AugmentedIRT("1pl", augmented=False)
    r = m.fit(X)
    lz = m.person_fit_lz()
    return {"loglik": r.loglik, "aic": r.aic, "bic": r.bic, "rmse": m.rmse(),
            "pct_fit": float(np.mean(lz >= -1.96) * 100), "n_params": r.n_params}


def compare(X, conf, model="2pl", n_options=4):
    """Fit both models on the same data and return everything Table 10 needs."""
    std = AugmentedIRT(model, augmented=False, n_options=n_options)
    aug = AugmentedIRT(model, augmented=True, n_options=n_options)
    rs, ra = std.fit(X), aug.fit(X, conf)
    lz_s, lz_a = std.person_fit_lz(), aug.person_fit_lz()
    return {
        "standard": {"loglik": rs.loglik, "aic": rs.aic, "bic": rs.bic, "rmse": std.rmse(),
                     "pct_fit": float(np.mean(lz_s >= -1.96) * 100), "n_params": rs.n_params},
        "augmented": {"loglik": ra.loglik, "aic": ra.aic, "bic": ra.bic, "rmse": aug.rmse(),
                      "pct_fit": float(np.mean(lz_a >= -1.96) * 100), "n_params": ra.n_params,
                      "lambda": ra.lam, "lambda_se": ra.lam_se},
        "lr_test": likelihood_ratio_test(rs, ra),
        "models": (std, aug),
    }

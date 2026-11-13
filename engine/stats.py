"""Statistical tests used to decide the hypotheses."""
from __future__ import annotations

import numpy as np
from scipy import stats


def corrected_resampled_ttest(a, b, n_train: float, n_test: float) -> dict:
    """Nadeau-Bengio corrected resampled t-test for paired cross-validation results.

    a, b: metric values of two models on the same folds (b - a is the improvement).
    The usual paired t-test is too optimistic because training folds overlap; the
    correction inflates the variance by (1/J + n_test/n_train).
    """
    d = np.asarray(b, float) - np.asarray(a, float)
    J = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return {"t": float("inf") if d.mean() else 0.0, "df": J - 1, "p": 0.0 if d.mean() else 1.0,
                "mean_diff": float(d.mean())}
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), J - 1)
    return {"t": float(t), "df": J - 1, "p": float(p), "mean_diff": float(d.mean())}


def paired_test(a, b) -> dict:
    """Paired t-test, or Wilcoxon signed-rank if the differences are not normal (Shapiro p < .05)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    normal = len(d) < 3 or stats.shapiro(d).pvalue >= 0.05
    if normal:
        r = stats.ttest_rel(b, a)
        sd = d.std(ddof=1)
        return {"test": "paired t-test", "statistic": float(r.statistic), "p": float(r.pvalue),
                "mean_diff": float(d.mean()), "cohens_d": float(d.mean() / sd) if sd else float("nan")}
    r = stats.wilcoxon(b, a)
    return {"test": "Wilcoxon signed-rank", "statistic": float(r.statistic), "p": float(r.pvalue),
            "mean_diff": float(d.mean()), "cohens_d": float("nan")}


def kr20(X) -> float:
    """Kuder-Richardson 20 reliability for a 0/1 response matrix (persons x items)."""
    X = np.asarray(X, float)
    k = X.shape[1]
    p = X.mean(0)
    var_total = X.sum(1).var(ddof=1)
    return float(k / (k - 1) * (1 - (p * (1 - p)).sum() / var_total)) if var_total > 0 else float("nan")

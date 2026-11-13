"""Comprehension prediction: Logistic Regression (baseline) vs Random Forest with SHAP."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["theta", "mastery_final", "mastery_gain", "confidence_avg", "trust_avg",
            "complexity_gap", "prior_knowledge", "year_level", "ai_courses"]
RF_PARAMS = dict(n_estimators=100, max_depth=10, min_samples_split=5)


def build_features(responses: pd.DataFrame, users: pd.DataFrame, theta: pd.Series,
                   mastery: pd.DataFrame, threshold: float = 0.70) -> tuple[pd.DataFrame, pd.Series]:
    """One row per user.  Every feature comes from data collected BEFORE the outcome.

    theta:   Series indexed by user_id (IRT ability)
    mastery: BKT predictions with columns user_id, skill, order, mastery
    target:  1 if users.post_score >= threshold
    """
    g = responses.groupby("user_id")
    feats = pd.DataFrame({"confidence_avg": g["confidence"].mean(), "trust_avg": g["trust"].mean()})
    acc = responses.pivot_table(index="user_id", columns="complexity", values="correct", aggfunc="mean")
    feats["complexity_gap"] = acc.get("L", 0) - acc.get("H", 0)
    m = mastery.sort_values("order").groupby(["user_id", "skill"])["mastery"]
    per_skill = pd.DataFrame({"final": m.last(), "first": m.first()}).groupby("user_id").mean()
    feats["mastery_final"] = per_skill["final"]
    feats["mastery_gain"] = per_skill["final"] - per_skill["first"]
    feats["theta"] = theta
    u = users.set_index("user_id")
    for col in ("prior_knowledge", "year_level", "ai_courses"):
        feats[col] = u[col]
    feats = feats[FEATURES].fillna(0.0)
    y = (u.loc[feats.index, "post_score"] >= threshold).astype(int)
    return feats, y


def _models(seed):
    return {"logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            "random_forest": RandomForestClassifier(random_state=seed, **RF_PARAMS)}


def cross_validate(X: pd.DataFrame, y: pd.Series, n_splits=5, n_repeats=10, seed=0) -> pd.DataFrame:
    """Both models on the SAME folds; one row per model per fold."""
    rows = []
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for i, (tr, te) in enumerate(cv.split(X, y)):
        for name, model in _models(seed + i).items():
            model.fit(X.iloc[tr], y.iloc[tr])
            prob = model.predict_proba(X.iloc[te])[:, 1]
            pred = (prob >= 0.5).astype(int)
            yt = y.iloc[te]
            rows.append({"fold": i, "model": name, "n_train": len(tr), "n_test": len(te),
                         "accuracy": accuracy_score(yt, pred),
                         "precision": precision_score(yt, pred, zero_division=0),
                         "recall": recall_score(yt, pred, zero_division=0),
                         "f1": f1_score(yt, pred, zero_division=0),
                         "roc_auc": roc_auc_score(yt, prob) if yt.nunique() > 1 else np.nan})
    return pd.DataFrame(rows)


def shap_ranking(X: pd.DataFrame, y: pd.Series, seed=0):
    """Fit the Random Forest on all data and rank features by mean |SHAP value|."""
    import shap
    rf = RandomForestClassifier(random_state=seed, **RF_PARAMS).fit(X, y)
    sv = shap.TreeExplainer(rf).shap_values(X)
    sv = np.asarray(sv)
    if sv.ndim == 3:            # (n, features, classes) in recent shap versions
        sv = sv[:, :, 1] if sv.shape[2] == 2 else sv[1]
    elif isinstance(sv, list):
        sv = sv[1]
    imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
    direction = pd.Series([np.corrcoef(X[c], sv[:, k])[0, 1] if X[c].std() > 0 else 0.0
                           for k, c in enumerate(X.columns)], index=X.columns)
    out = pd.DataFrame({"mean_abs_shap": imp, "direction": direction.loc[imp.index]})
    out["rank"] = range(1, len(out) + 1)
    return out, rf, sv

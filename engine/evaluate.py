"""Run the full algorithm evaluation (H1-H5 plus baselines) on the data the app collected.

H5 is the ablation study: the Random Forest is trained four times with the same folds,
using ability (theta) from the standard or augmented IRT and mastery from the standard or
augmented BKT, to show how much each augmentation adds on its own and together.
Baselines: Rasch (1PL) IRT and BKT with forgetting.

Input (the app's export, or the synthetic generator):
  responses: user_id, session, order, item_id, skill, complexity (L/H), correct (0/1),
             confidence (1-5), trust (1-5)
  users:     user_id, prior_knowledge, year_level, ai_courses, post_score (0-1)

Output: a dict with every number the Chapter 4 tables need, plus a Markdown report.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import bkt as bkt_mod
from . import irt as irt_mod
from . import predict as pred_mod
from .stats import corrected_resampled_ttest, kr20

KEY_FEATURES = ("prior_knowledge", "complexity_gap", "confidence_avg")
ABLATION = (("none", "standard IRT + standard BKT", False, False),
            ("lambda_only", "augmented IRT (λ) + standard BKT", True, False),
            ("trust_only", "standard IRT + augmented BKT (trust)", False, True),
            ("full", "augmented IRT (λ) + augmented BKT (trust)", True, True))


def ablation(responses, users, theta_std, theta_aug, n_splits=5, n_repeats=10, seed=0, alpha=0.05):
    """H5: Random Forest with each combination of standard/augmented components, same folds."""
    mastery = {False: bkt_mod.BKT(augmented=False).fit(responses).predict(responses),
               True: bkt_mod.BKT(augmented=True).fit(responses).predict(responses)}
    rows, folds = [], {}
    for key, label, use_lam, use_trust in ABLATION:
        Xf, y = pred_mod.build_features(responses, users, theta_aug if use_lam else theta_std, mastery[use_trust])
        cv = pred_mod.cross_validate(Xf, y, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
        cv = cv[cv["model"] == "random_forest"]
        folds[key] = cv.set_index("fold")
        rows.append({"config": key, "label": label, **cv[["accuracy", "f1", "roc_auc"]].mean().to_dict()})
    tab = pd.DataFrame(rows).set_index("config")
    n_tr, n_te = folds["full"]["n_train"].mean(), folds["full"]["n_test"].mean()
    tests = {k: corrected_resampled_ttest(folds[k]["roc_auc"], folds["full"]["roc_auc"], n_tr, n_te)
             for k in ("none", "lambda_only", "trust_only")}
    for k, t in tests.items():
        tab.loc[k, "auc_gain_of_full"] = tab.loc["full", "roc_auc"] - tab.loc[k, "roc_auc"]
        tab.loc[k, "p_vs_full"] = t["p"]
    best = tab["roc_auc"].idxmax() == "full"
    supported = bool(best and tests["none"]["p"] < alpha and tab.loc["none", "auc_gain_of_full"] > 0)
    return {"table": tab.reset_index().to_dict("records"), "tests": tests,
            "contribution": {"lambda": float(tab.loc["lambda_only", "roc_auc"] - tab.loc["none", "roc_auc"]),
                             "trust": float(tab.loc["trust_only", "roc_auc"] - tab.loc["none", "roc_auc"])},
            "supported": supported}


def response_matrices(responses: pd.DataFrame):
    first = responses.sort_values("order").drop_duplicates(["user_id", "item_id"])
    X = first.pivot(index="user_id", columns="item_id", values="correct")
    C = first.pivot(index="user_id", columns="item_id", values="confidence").reindex_like(X)
    return X, C


def run(responses: pd.DataFrame, users: pd.DataFrame, irt_model="2pl", n_options=4,
        n_splits=5, n_repeats=10, alpha=0.05, f1_rel_gain=0.10, auc_gain=0.03, seed=0) -> dict:
    t0 = time.time()
    out: dict = {"n_users": int(responses["user_id"].nunique()), "n_responses": int(len(responses))}

    # ---------------- descriptive
    X, C = response_matrices(responses)
    out["descriptive"] = {
        "mean_pct_correct": float(X.mean(axis=1).mean() * 100), "sd_pct_correct": float(X.mean(axis=1).std() * 100),
        "kr20": kr20(X.fillna(0).to_numpy()),
        "by_type": responses.groupby("skill")["correct"].mean().mul(100).round(1).to_dict(),
        "by_complexity": responses.groupby("complexity")["correct"].mean().mul(100).round(1).to_dict(),
        "confidence_by_type": responses.groupby("skill")["confidence"].mean().round(2).to_dict(),
        "trust_by_type": responses.groupby("skill")["trust"].mean().round(2).to_dict(),
    }

    # ---------------- H1: IRT
    irt = irt_mod.compare(X.to_numpy(float), C.to_numpy(float), model=irt_model, n_options=n_options)
    std_irt, aug_irt = irt.pop("models")
    theta = pd.Series(aug_irt.result.theta, index=X.index)
    theta_std = pd.Series(std_irt.result.theta, index=X.index)
    h1 = irt["lr_test"]["p"] < alpha and irt["augmented"]["aic"] < irt["standard"]["aic"]
    out["H1_irt"] = {**irt, "supported": bool(h1)}
    out["baseline_irt_rasch"] = irt_mod.baseline_rasch(X.to_numpy(float))

    # ---------------- H2: BKT
    cv_b = bkt_mod.cross_validate(responses, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    piv = cv_b.pivot_table(index=["repeat", "fold"], columns="model", values="auc")
    n_tr, n_te = cv_b["n_train"].mean(), cv_b["n_test"].mean()
    t_b = corrected_resampled_ttest(piv["standard"], piv["augmented"], n_tr, n_te)
    summary_b = cv_b.groupby("model")[["auc", "rmse", "accuracy"]].mean()
    full_bkt = bkt_mod.BKT(augmented=True).fit(responses)
    gain = summary_b.loc["augmented", "auc"] - summary_b.loc["standard", "auc"]
    h2 = gain >= auc_gain and t_b["p"] < alpha and gain > 0
    pf_ = cv_b.pivot_table(index=["repeat", "fold"], columns="model", values="auc")
    out["baseline_bkt_forgetting"] = {**summary_b.loc["forgetting"].to_dict(),
                                      "test_vs_augmented": corrected_resampled_ttest(pf_["forgetting"], pf_["augmented"], n_tr, n_te),
                                      "params": {s_: p.as_dict() for s_, p in
                                                 bkt_mod.BKT(augmented=False, forgetting=True).fit(responses).params.items()}}
    out["H2_bkt"] = {"standard": summary_b.loc["standard"].to_dict(), "augmented": summary_b.loc["augmented"].to_dict(),
                     "auc_gain": float(gain), "test": t_b,
                     "params": {s: p.as_dict() for s, p in full_bkt.params.items()},
                     "lr_test_trust_rates": bkt_mod.likelihood_ratio_test(responses), "supported": bool(h2)}

    # ---------------- H3 / H4: prediction
    mastery = full_bkt.predict(responses)
    Xf, y = pred_mod.build_features(responses, users, theta, mastery)
    out["class_balance"] = {"proficient": int(y.sum()), "needs_improvement": int((1 - y).sum())}
    cv_p = pred_mod.cross_validate(Xf, y, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    summ = cv_p.groupby("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]].mean()
    pf = cv_p.pivot_table(index="fold", columns="model", values="f1")
    t_p = corrected_resampled_ttest(pf["logistic_regression"], pf["random_forest"],
                                    cv_p["n_train"].mean(), cv_p["n_test"].mean())
    lr_f1, rf_f1 = summ.loc["logistic_regression", "f1"], summ.loc["random_forest", "f1"]
    rel = (rf_f1 - lr_f1) / lr_f1 if lr_f1 > 0 else float("nan")
    h3 = rel >= f1_rel_gain and t_p["p"] < alpha
    out["H3_prediction"] = {"logistic_regression": summ.loc["logistic_regression"].to_dict(),
                            "random_forest": summ.loc["random_forest"].to_dict(),
                            "f1_relative_gain": float(rel), "test": t_p, "supported": bool(h3)}
    ranking, _, _ = pred_mod.shap_ranking(Xf, y, seed=seed)
    top5 = list(ranking.index[:5])
    out["H4_shap"] = {"ranking": ranking.round(4).reset_index(names="feature").to_dict("records"),
                      "top5": top5, "key_features_in_top5": {f: f in top5 for f in KEY_FEATURES},
                      "supported": all(f in top5 for f in KEY_FEATURES)}
    out["H5_ablation"] = ablation(responses, users, theta_std, theta, n_splits, n_repeats, seed, alpha)
    out["runtime_seconds"] = round(time.time() - t0, 1)
    return out


def report_markdown(r: dict) -> str:
    f = lambda v, d=3: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"
    yes = lambda b: "Supported" if b else "Not supported"
    h1, h2, h3, h4 = r["H1_irt"], r["H2_bkt"], r["H3_prediction"], r["H4_shap"]
    lines = [f"# XAI-Check evaluation report", "",
             f"Users: {r['n_users']} · Responses: {r['n_responses']} · Runtime: {r['runtime_seconds']} s", "",
             "## H1 – Standard vs augmented IRT", "",
             "| Metric | Standard | Augmented |", "|---|---|---|",
             f"| Log-likelihood | {f(h1['standard']['loglik'],1)} | {f(h1['augmented']['loglik'],1)} |",
             f"| AIC | {f(h1['standard']['aic'],1)} | {f(h1['augmented']['aic'],1)} |",
             f"| BIC | {f(h1['standard']['bic'],1)} | {f(h1['augmented']['bic'],1)} |",
             f"| RMSE | {f(h1['standard']['rmse'])} | {f(h1['augmented']['rmse'])} |",
             f"| Persons with lz ≥ −1.96 (%) | {f(h1['standard']['pct_fit'],1)} | {f(h1['augmented']['pct_fit'],1)} |",
             f"| λ (SE) | — | {f(h1['augmented']['lambda'])} ({f(h1['augmented']['lambda_se'])}) |", "",
             f"Likelihood-ratio test: χ²({h1['lr_test']['df']}) = {f(h1['lr_test']['lr'],2)}, p = {f(h1['lr_test']['p'],4)} → **{yes(h1['supported'])}**", "",
             "## H2 – Standard vs augmented BKT (cross-validated)", "",
             "| Metric | Standard | Augmented |", "|---|---|---|"]
    for k in ("auc", "rmse", "accuracy"):
        lines.append(f"| {k.upper()} | {f(h2['standard'][k])} | {f(h2['augmented'][k])} |")
    lines += ["", f"AUC gain = {f(h2['auc_gain'])}; corrected t({h2['test']['df']}) = {f(h2['test']['t'],2)}, "
                  f"p = {f(h2['test']['p'],4)} → **{yes(h2['supported'])}**", "",
              f"Do the trust-level learning rates differ? LR χ²({h2['lr_test_trust_rates']['df']}) = "
              f"{f(h2['lr_test_trust_rates']['lr'],2)}, p = {f(h2['lr_test_trust_rates']['p'],4)}", "",
              "| Skill | P(L0) | P(T\\|low) | P(T\\|medium) | P(T\\|high) | P(G) | P(S) |", "|---|---|---|---|---|---|---|"]
    for s, p in h2["params"].items():
        lines.append(f"| {s} | {f(p['L0'])} | {f(p['T_low'])} | {f(p['T_medium'])} | {f(p['T_high'])} | {f(p['G'])} | {f(p['S'])} |")
    lines += ["", "## H3 – Logistic Regression vs Random Forest", "",
              "| Metric | Logistic Regression | Random Forest |", "|---|---|---|"]
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        lines.append(f"| {k} | {f(h3['logistic_regression'][k])} | {f(h3['random_forest'][k])} |")
    lines += ["", f"F1 relative gain = {f(h3['f1_relative_gain']*100,1)}%; corrected t({h3['test']['df']}) = "
                  f"{f(h3['test']['t'],2)}, p = {f(h3['test']['p'],4)} → **{yes(h3['supported'])}**", "",
              "## H4 – SHAP feature ranking", "", "| Rank | Feature | Mean \\|SHAP\\| | Direction |", "|---|---|---|---|"]
    for row in h4["ranking"]:
        lines.append(f"| {row['rank']} | {row['feature']} | {f(row['mean_abs_shap'],4)} | {'+' if row['direction'] >= 0 else '−'} |")
    lines += ["", f"Top five: {', '.join(h4['top5'])} → **{yes(h4['supported'])}**", ""]
    h5 = r["H5_ablation"]
    lines += ["## H5 – Ablation study (Random Forest, same folds)", "",
              "| Configuration | Accuracy | F1 | ROC-AUC | AUC gain of full model | p (vs full) |", "|---|---|---|---|---|---|"]
    for row in h5["table"]:
        lines.append(f"| {row['label']} | {f(row['accuracy'])} | {f(row['f1'])} | {f(row['roc_auc'])} | "
                     f"{f(row.get('auc_gain_of_full'))} | {f(row.get('p_vs_full'),4)} |")
    lines += ["", f"Contribution of λ alone: {f(h5['contribution']['lambda'])} AUC; trust alone: "
                  f"{f(h5['contribution']['trust'])} AUC → **{yes(h5['supported'])}**", "",
              "## Additional baselines", "",
              "| Model | Log-likelihood | AIC | BIC |", "|---|---|---|---|"]
    ra = r["baseline_irt_rasch"]
    lines.append(f"| Rasch (1PL) IRT | {f(ra['loglik'],1)} | {f(ra['aic'],1)} | {f(ra['bic'],1)} |")
    lines.append(f"| Standard 2PL IRT | {f(h1['standard']['loglik'],1)} | {f(h1['standard']['aic'],1)} | {f(h1['standard']['bic'],1)} |")
    lines.append(f"| Augmented 2PL IRT (λ) | {f(h1['augmented']['loglik'],1)} | {f(h1['augmented']['aic'],1)} | {f(h1['augmented']['bic'],1)} |")
    bf = r["baseline_bkt_forgetting"]
    lines += ["", "| Model | AUC | RMSE | Accuracy |", "|---|---|---|---|",
              f"| Standard BKT | {f(h2['standard']['auc'])} | {f(h2['standard']['rmse'])} | {f(h2['standard']['accuracy'])} |",
              f"| BKT with forgetting | {f(bf['auc'])} | {f(bf['rmse'])} | {f(bf['accuracy'])} |",
              f"| Augmented BKT (trust) | {f(h2['augmented']['auc'])} | {f(h2['augmented']['rmse'])} | {f(h2['augmented']['accuracy'])} |",
              "", f"Augmented vs forgetting: corrected t = {f(bf['test_vs_augmented']['t'],2)}, p = {f(bf['test_vs_augmented']['p'],4)}", ""]
    return "\n".join(lines)


def save(r: dict, path_json: str, path_md: str):
    with open(path_json, "w") as fh:
        json.dump(r, fh, indent=2, default=float)
    with open(path_md, "w") as fh:
        fh.write(report_markdown(r))

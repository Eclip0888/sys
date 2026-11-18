"""Live scoring with the evaluation engine, and calibration from the collected data.

Before calibration the engine uses neutral defaults (a = 1, b = 0, lambda = 0 and textbook
BKT values).  After enough users finish, the admin presses "Calibrate" and the augmented IRT,
augmented BKT and Random Forest are fitted on the real responses; from then on every new
answer is scored with the calibrated values.
"""
from __future__ import annotations

import io
import math

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from engine import bkt as bkt_mod
from engine import predict as pred_mod
from engine.irt import AugmentedIRT, _quadrature, center_confidence

from .models import Check, Param, User, db
from .questions import PREFIX, TYPES

ITEMS = [f"{PREFIX[k]}{n}" for k in TYPES for n in range(1, 9)]
DEFAULT_BKT = {"L0": 0.3, "T": [0.1, 0.1, 0.1], "G": 0.2, "S": 0.1}
MIN_USERS_TO_CALIBRATE = 30


def _param(key, default=None):
    p = db.session.get(Param, key)
    return p if p is not None else default


def _save(key, value=None, blob=None):
    p = db.session.get(Param, key) or Param(key=key)
    p.value, p.blob = value, blob
    db.session.add(p)
    db.session.commit()


def calibration_status():
    p = _param("irt")
    return {"calibrated": p is not None, "updated_at": p.updated_at if p else None,
            "n_users": (p.value or {}).get("n_users") if p else None}


# ------------------------------------------------------------------ live scoring
def ability(checks: list[Check]) -> tuple[float, float]:
    """EAP ability (theta) and its SE from the main checks answered so far."""
    p = _param("irt")
    v = p.value if p else {"a": {}, "b": {}, "lam": 0.0}
    done = [c for c in checks if c.phase == "main" and c.correct is not None]
    if not done:
        return 0.0, 1.0
    a = np.array([v["a"].get(c.item_id, 1.0) for c in done])
    b = np.array([v["b"].get(c.item_id, 0.0) for c in done])
    x = np.array([c.correct for c in done], float)[None]
    conf = np.array([c.confidence or 3 for c in done], float)[None]
    m = center_confidence(conf, np.ones_like(conf, bool))
    nodes, w = _quadrature()
    P = np.clip(expit(a * (nodes[:, None, None] - b) + v["lam"] * m[None]), 1e-9, 1 - 1e-9)
    ll = (x * np.log(P) + (1 - x) * np.log(1 - P)).sum(axis=(1, 2)) + np.log(w)
    post = np.exp(ll - ll.max()); post /= post.sum()
    mean = float((post * nodes).sum())
    return mean, float(math.sqrt((post * (nodes - mean) ** 2).sum()))


def mastery(checks: list[Check]) -> dict:
    """Augmented-BKT mastery per explanation type after the main checks so far."""
    p = _param("bkt")
    params = p.value if p else {k: DEFAULT_BKT for k in TYPES}
    out = {}
    for k in TYPES:
        q = params.get(k, DEFAULT_BKT)
        L = q["L0"]
        for c in sorted((c for c in checks if c.phase == "main" and c.skill == k and c.correct is not None),
                        key=lambda c: c.order):
            if c.correct:
                post = L * (1 - q["S"]) / (L * (1 - q["S"]) + (1 - L) * q["G"])
            else:
                post = L * q["S"] / (L * q["S"] + (1 - L) * (1 - q["G"]))
            lvl = 0 if (c.trust or 3) <= 2 else (1 if c.trust == 3 else 2)
            L = post * (1 - q.get("F", 0.0)) + (1 - post) * q["T"][lvl]
        out[k] = float(L)
    return out


def predicted_comprehension(user: User, checks: list[Check]):
    """Random Forest prediction of proficiency (needs calibration); returns (prob, top factors) or None."""
    p = _param("rf")
    if p is None or p.blob is None:
        return None
    rf = joblib.load(io.BytesIO(p.blob))
    resp = responses_frame([c for c in checks if c.phase == "main"], {user.id: user})
    if resp.empty:
        return None
    th, _ = ability(checks)
    m = bkt_mod.BKT(augmented=True)
    m.params = {k: bkt_mod.SkillParams(v["L0"], np.array(v["T"]), v["G"], v["S"], v.get("F", 0.0))
                for k, v in (_param("bkt").value.items())}
    users = users_frame([user], {user.id: 0.0})
    X, _ = pred_mod.build_features(resp, users, pd.Series({user.code: th}), m.predict(resp))
    prob = float(rf.predict_proba(X)[0, 1])
    return prob


# ------------------------------------------------------------------ data frames (engine format)
def responses_frame(checks, users_by_id) -> pd.DataFrame:
    rows = []
    for c in checks:
        if c.correct is None:
            continue
        u = users_by_id[c.user_id]
        rows.append({"user_id": u.code, "session": (c.order - 1) // 8 + 1 if c.phase == "main" else 0,
                     "order": c.order, "item_id": c.item_id, "skill": c.skill, "complexity": c.complexity,
                     "correct": c.correct, "confidence": c.confidence, "trust": c.trust,
                     "phase": c.phase, "answered_at": c.answered_at})
    return pd.DataFrame(rows)


def users_frame(users, post_override=None) -> pd.DataFrame:
    return pd.DataFrame([{
        "user_id": u.code, "prior_knowledge": u.prior_score or 0, "year_level": u.year_level or 0,
        "ai_courses": u.ai_courses or 0,
        "post_score": (post_override or {}).get(u.id, u.post_score if u.post_score is not None else np.nan),
        "prog_courses": u.prog_courses,
        "completed": u.stage == "done", "excluded": bool(u.excluded),
        "started_at": u.created_at, "finished_at": u.finished_at} for u in users])


def study_data(include_unfinished=False):
    users = User.query.filter_by(excluded=False).all()
    if not include_unfinished:
        users = [u for u in users if u.stage == "done"]
    by_id = {u.id: u for u in users}
    checks = Check.query.filter(Check.user_id.in_(list(by_id) or [-1])).all()
    resp = responses_frame(checks, by_id)
    return users, resp


# ------------------------------------------------------------------ calibration
def calibrate() -> dict:
    users, resp = study_data()
    if len(users) < MIN_USERS_TO_CALIBRATE:
        return {"ok": False, "message": f"Need at least {MIN_USERS_TO_CALIBRATE} completed users (have {len(users)})."}
    main = resp[resp["phase"] == "main"]
    X = main.pivot_table(index="user_id", columns="item_id", values="correct").reindex(columns=ITEMS)
    C = main.pivot_table(index="user_id", columns="item_id", values="confidence").reindex_like(X)
    irt = AugmentedIRT("2pl", augmented=True)
    r = irt.fit(X.to_numpy(float), C.to_numpy(float))
    _save("irt", {"a": dict(zip(ITEMS, map(float, r.a))), "b": dict(zip(ITEMS, map(float, r.b))),
                  "lam": float(r.lam), "lam_se": float(r.lam_se), "n_users": len(users)})
    bk = bkt_mod.BKT(augmented=True).fit(main)
    _save("bkt", {k: {"L0": p.L0, "T": [float(t) for t in p.T], "G": p.G, "S": p.S, "F": p.F}
                  for k, p in bk.params.items()})
    theta = pd.Series(r.theta, index=X.index)
    feats, y = pred_mod.build_features(main, users_frame(users), theta, bk.predict(main))
    msg = f"Calibrated on {len(users)} users: λ = {r.lam:.2f}."
    if y.nunique() == 2:
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(random_state=0, **pred_mod.RF_PARAMS).fit(feats, y)
        buf = io.BytesIO(); joblib.dump(rf, buf)
        _save("rf", {"features": list(feats.columns), "n": len(y)}, buf.getvalue())
        msg += " Random Forest trained."
    else:
        msg += " Random Forest not trained yet (all users in one class)."
    return {"ok": True, "message": msg}

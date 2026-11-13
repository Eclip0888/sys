"""The AI that users talk to: an academic-performance model with three kinds of explanation.

* Prediction model: Random Forest (scikit-learn) that predicts PASS / FAIL from six study habits.
  It is trained on a synthetic dataset generated with a known rule (see `make_dataset`).
  Replace `make_dataset` with a real dataset (for example the UCI Student Performance data)
  if your adviser asks for one; nothing else needs to change.
* Explanations: SHAP (TreeExplainer), LIME (LimeTabularExplainer) and a counterfactual
  found by searching the smallest change to the actionable factors that flips the prediction.
* Parser: reads the numbers in a free-text question ("Will I pass if I study 6 hours a week
  and have 3 absences?") and maps them to the model's inputs.
"""
from __future__ import annotations

import itertools
import os
import re
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------------ features
@dataclass(frozen=True)
class Feature:
    key: str
    label: str          # shown to users
    unit: str
    lo: float
    hi: float
    default: float
    step: float
    actionable: bool    # can the student change it now?
    keywords: tuple


FEATURES = (
    Feature("study_hours", "Study hours per week", "h/week", 0, 30, 8, 1, True,
            ("study", "studies", "studying", "review", "reviewing", "aral")),
    Feature("absences", "Absences this term", "absences", 0, 20, 3, 1, True,
            ("absence", "absences", "absent", "miss", "missed", "lliban", "liban")),
    Feature("sleep_hours", "Sleep per night", "h/night", 3, 10, 7, 0.5, True,
            ("sleep", "sleeping", "tulog")),
    Feature("assignments_done", "Assignments submitted", "%", 0, 100, 80, 5, True,
            ("assignment", "assignments", "homework", "requirements", "activities", "submitted", "submit")),
    Feature("extracurricular_hours", "Extracurricular / work hours per week", "h/week", 0, 25, 5, 1, True,
            ("extracurricular", "org", "organization", "work", "job", "part-time", "parttime", "sports")),
    Feature("prev_grade", "Previous term grade", "%", 60, 100, 82, 1, False,
            ("grade", "gwa", "average", "previous", "last")),
)
KEYS = [f.key for f in FEATURES]
F = {f.key: f for f in FEATURES}
CLASSES = ("FAIL", "PASS")


def make_dataset(n=4000, seed=42) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic students with a known, plausible rule (documented in the thesis)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "study_hours": np.clip(rng.gamma(3.0, 3.0, n), 0, 30).round(0),
        "absences": np.clip(rng.poisson(3.5, n), 0, 20).astype(float),
        "sleep_hours": np.clip(rng.normal(6.8, 1.1, n), 3, 10).round(1),
        "assignments_done": np.clip(rng.normal(78, 15, n), 0, 100).round(0),
        "extracurricular_hours": np.clip(rng.gamma(1.6, 3.5, n), 0, 25).round(0),
        "prev_grade": np.clip(rng.normal(82, 7, n), 60, 100).round(0),
    })
    z = (0.17 * (X.study_hours - 8) - 0.30 * (X.absences - 3.5) + 0.35 * (X.sleep_hours - 6.8)
         + 0.04 * (X.assignments_done - 78) - 0.09 * (X.extracurricular_hours - 5)
         + 0.13 * (X.prev_grade - 82) + 0.6 + rng.normal(0, 0.6, n))
    y = (z > 0).astype(int)
    return X[KEYS], y.to_numpy()


class Predictor:
    def __init__(self, cache_dir: str | None = None):
        cache = os.path.join(cache_dir or os.path.dirname(__file__), "model.joblib")
        if os.path.exists(cache):
            self.model, self.background = joblib.load(cache)
        else:
            X, y = make_dataset()
            self.model = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=5,
                                                random_state=0).fit(X, y)
            self.background = X.sample(100, random_state=0)
            try:
                joblib.dump((self.model, self.background), cache)
            except OSError:
                pass
        self._shap = None
        self._lime = None

    # ---------------------------------------------------------------- prediction
    def frame(self, inputs: dict) -> pd.DataFrame:
        return pd.DataFrame([[float(inputs[k]) for k in KEYS]], columns=KEYS)

    def prob_pass(self, inputs: dict) -> float:
        return float(self.model.predict_proba(self.frame(inputs))[0, 1])

    def predict(self, inputs: dict) -> dict:
        p = self.prob_pass(inputs)
        return {"label": CLASSES[int(p >= 0.5)], "prob_pass": p}

    # ---------------------------------------------------------------- SHAP
    def shap(self, inputs: dict) -> dict:
        import shap
        if self._shap is None:
            self._shap = shap.TreeExplainer(self.model, data=self.background, model_output="probability")
        sv = np.asarray(self._shap.shap_values(self.frame(inputs)))
        if sv.ndim == 3:
            sv = sv[0, :, 1]
        elif sv.ndim == 2:
            sv = sv[0]
        base = self._shap.expected_value
        base = float(np.atleast_1d(base)[-1])
        contrib = {k: float(v) for k, v in zip(KEYS, sv)}
        return {"type": "SHAP", "base": base, "prob_pass": self.prob_pass(inputs), "contrib": contrib}

    # ---------------------------------------------------------------- LIME
    def lime(self, inputs: dict, seed: int = 0) -> dict:
        from lime.lime_tabular import LimeTabularExplainer
        if self._lime is None:
            self._lime = LimeTabularExplainer(self.background.to_numpy(), feature_names=KEYS,
                                              class_names=list(CLASSES), discretize_continuous=True,
                                              mode="classification", random_state=seed)
        x = self.frame(inputs).to_numpy()[0]
        fn = lambda a: self.model.predict_proba(pd.DataFrame(a, columns=KEYS))
        exp = self._lime.explain_instance(x, fn, num_features=4, num_samples=1500, labels=(1,))
        conds = []
        for rule, w in exp.as_list(label=1):
            key = next(k for k in sorted(KEYS, key=len, reverse=True) if k in rule)
            conds.append({"rule": _pretty_rule(rule, key), "feature": key, "weight": float(w)})
        return {"type": "LIME", "prob_pass": self.prob_pass(inputs),
                "local_pred": float(exp.local_pred[0]) if exp.local_pred is not None else None,
                "intercept": float(exp.intercept[1]), "conditions": conds}

    # ---------------------------------------------------------------- counterfactual
    def counterfactual(self, inputs: dict, max_changes: int = 3) -> dict:
        """Smallest change to the actionable factors that flips the prediction (DiCE-style search)."""
        p0 = self.prob_pass(inputs)
        want_pass = p0 < 0.5
        act = [f for f in FEATURES if f.actionable]

        def candidates(f):
            cur = float(inputs[f.key])
            vals = np.arange(f.lo, f.hi + 1e-9, f.step)
            vals = vals[np.abs(vals - cur) > 1e-9]
            return sorted(vals, key=lambda v: abs(v - cur))

        best = None
        for r in range(1, max_changes + 1):
            for combo in itertools.combinations(act, r):
                grids = [candidates(f)[:(40 if r < 3 else 12)] for f in combo]
                rows, changes = [], []
                for vals in itertools.product(*grids):
                    x = dict(inputs)
                    for f, v in zip(combo, vals):
                        x[f.key] = float(v)
                    rows.append([x[k] for k in KEYS])
                    changes.append(vals)
                if not rows:
                    continue
                probs = self.model.predict_proba(pd.DataFrame(rows, columns=KEYS))[:, 1]
                for vals, p in zip(changes, probs):
                    if (p >= 0.5) != want_pass:
                        continue
                    cost = sum(abs(v - float(inputs[f.key])) / (f.hi - f.lo) for f, v in zip(combo, vals))
                    if best is None or cost < best[0] - 1e-12:
                        best = (cost, combo, vals, float(p))
            if best is not None:
                break
        if best is None:
            best = self._greedy_flip(inputs, want_pass)
        if best is None:
            return {"type": "CF", "found": False, "prob_pass": p0, "label": CLASSES[int(p0 >= 0.5)]}
        _, combo, vals, p1 = best
        new = dict(inputs)
        for f, v in zip(combo, vals):
            new[f.key] = float(v)
        return {"type": "CF", "found": True, "prob_pass": p0, "label": CLASSES[int(p0 >= 0.5)],
                "new_label": CLASSES[int(p1 >= 0.5)], "new_prob": p1, "new_inputs": new,
                "changes": [{"feature": f.key, "old": float(inputs[f.key]), "new": float(v)} for f, v in zip(combo, vals)]}

    def _greedy_flip(self, inputs, want_pass, max_steps=40):
        """Fallback: repeatedly take the single step that moves the probability most toward the flip."""
        x = dict(inputs)
        act = [f for f in FEATURES if f.actionable]
        for _ in range(max_steps):
            rows, moves = [], []
            for f in act:
                for d in (-1, 1):
                    v = min(max(x[f.key] + d * max(f.step, (f.hi - f.lo) / 10), f.lo), f.hi)
                    if v != x[f.key]:
                        y = dict(x); y[f.key] = v
                        rows.append([y[k] for k in KEYS]); moves.append((f.key, v))
            probs = self.model.predict_proba(pd.DataFrame(rows, columns=KEYS))[:, 1]
            k = int(np.argmax(probs) if want_pass else np.argmin(probs))
            x[moves[k][0]] = moves[k][1]
            if (probs[k] >= 0.5) == want_pass:
                combo = tuple(F[key] for key in KEYS if x[key] != inputs[key])
                return (0.0, combo, tuple(x[f.key] for f in combo), float(probs[k]))
        return None

    def explain(self, kind: str, inputs: dict) -> dict:
        return {"SHAP": self.shap, "LIME": self.lime, "CF": self.counterfactual}[kind](inputs)


def _pretty_rule(rule: str, key: str) -> str:
    """'absences > 4.00' -> 'Absences this term > 4'."""
    def num(m):
        v = float(m.group(0))
        return str(int(v)) if abs(v - round(v)) < 1e-6 else f"{v:.1f}"
    return re.sub(r"-?\d+\.\d+", num, rule).replace(key, F[key].label)


def fmt(key: str, v: float) -> str:
    f = F[key]
    s = str(int(v)) if abs(v - round(v)) < 1e-6 else f"{v:.1f}"
    return f"{s}%" if f.unit == "%" else f"{s} {f.unit}"


# ------------------------------------------------------------------ question parser
WORDNUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve "
                                      "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
WORDNUM.update({"isa": 1, "dalawa": 2, "tatlo": 3, "apat": 4, "lima": 5, "anim": 6, "pito": 7, "walo": 8,
                "siyam": 9, "sampu": 10, "no": 0, "none": 0, "zero": 0, "half": 0.5})


def parse_question(text: str, defaults: dict | None = None) -> tuple[dict, list]:
    """Return (inputs, found_keys).  Numbers are matched to the nearest feature keyword."""
    inputs = {f.key: f.default for f in FEATURES}
    if defaults:
        inputs.update({k: v for k, v in defaults.items() if k in inputs})
    tokens = re.findall(r"[a-zA-Z\-]+|\d+(?:\.\d+)?%?", text.lower())
    nums = []
    for i, t in enumerate(tokens):
        if re.fullmatch(r"\d+(?:\.\d+)?%?", t):
            nums.append((i, float(t.rstrip("%"))))
        elif t in WORDNUM and t not in ("no", "none", "half"):
            nums.append((i, float(WORDNUM[t])))
        elif t in ("no", "none", "zero"):
            nums.append((i, 0.0))
    kw_pos = []
    for i, t in enumerate(tokens):
        for f in FEATURES:
            if any(t.startswith(k) for k in f.keywords):
                kw_pos.append((i, f.key))
    found, used = [], set()
    pairs = sorted(((abs(i - j) + (0.5 if j < i else 0), ni, key, v)
                    for ni, (i, v) in enumerate(nums) for j, key in kw_pos if abs(i - j) <= 6))
    for _, ni, key, v in pairs:
        if ni in used or key in found:
            continue
        f = F[key]
        if key == "prev_grade" and v <= 5:          # Philippine 1.0-5.0 scale -> percentage
            v = {1: 97, 1.25: 94, 1.5: 91, 1.75: 88, 2: 85, 2.25: 82, 2.5: 79, 2.75: 76, 3: 75}.get(v, 70)
        if key in ("study_hours", "extracurricular_hours") and re.search(r"(a|per|each|every)\s+day|daily|araw", text.lower()):
            v = v * 5
        inputs[key] = float(min(max(v, f.lo), f.hi))
        found.append(key)
        used.add(ni)
    return inputs, found

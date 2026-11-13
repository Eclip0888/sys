"""24 comprehension-check templates (8 per explanation type: 4 low and 4 high complexity).

Each template reads the explanation the user just saw and builds a multiple-choice question
whose correct answer is computed from that explanation.  The template ID is the IRT item:
every user answers each template once, so the 24 templates are calibrated (difficulty b,
discrimination a) on the collected responses.

A template returns {"prompt", "options" (list of str), "answer" (index of the correct option)}.
"""
from __future__ import annotations

import random

from .predictor import F, KEYS, fmt

TYPES = ("SHAP", "LIME", "CF")
TYPE_NAMES = {"SHAP": "SHAP", "LIME": "LIME", "CF": "Counterfactual"}


def _mcq(prompt, correct, distractors, rng, keep_order=False):
    seen, opts = {correct}, [correct]
    for d in distractors:
        if d not in seen:
            seen.add(d)
            opts.append(d)
        if len(opts) == 4:
            break
    rng.shuffle(opts)        # always shuffle so the correct answer is not in a fixed position
    return {"prompt": prompt, "options": opts, "answer": opts.index(correct)}


def _lab(k):
    return F[k].label


SHAP_TOL = 0.0015   # SHAP bars are shown in percentage points with 1 decimal
LIME_TOL = 0.006    # LIME weights are shown with 2 decimals


def _apart(values, ref, tol):
    """Keep only items whose magnitude differs visibly from the reference (no on-screen ties)."""
    return [k for k in values if abs(abs(values[k]) - abs(ref)) > tol]


def _pct(p):
    return f"{p * 100:.0f}%"


# ============================================================ SHAP
def S1(e, rng):
    c = e["contrib"]
    top = max(c, key=lambda k: abs(c[k]))
    others = [k for k in sorted(c, key=lambda k: -abs(c[k])) if k in _apart(c, c[top], SHAP_TOL)]
    return _mcq("Which factor had the BIGGEST effect on this prediction (the longest bar)?",
                _lab(top), [_lab(k) for k in others[:1] + rng.sample(others[1:], 2)], rng)


def S2(e, rng):
    c = e["contrib"]
    k = rng.choice([k for k in c if abs(c[k]) >= 0.01] or list(c))
    ans = "It raised the chance of passing" if c[k] > 0 else "It lowered the chance of passing"
    return _mcq(f"In this explanation, what did “{_lab(k)}” do?", ans,
                ["It raised the chance of passing", "It lowered the chance of passing",
                 "It had no effect at all", "It changed the student’s real grade"], rng, keep_order=True)


def S3(e, rng):
    c = e["contrib"]
    low = min(c, key=lambda k: abs(c[k]))
    others = [k for k in sorted(c, key=lambda k: -abs(c[k])) if k in _apart(c, c[low], SHAP_TOL)]
    return _mcq("Which factor had the SMALLEST effect on this prediction?",
                _lab(low), [_lab(k) for k in others[:3]], rng)


def S4(e, rng):
    return _mcq("What does this SHAP explanation describe?",
                "Why the model gave THIS prediction for these inputs",
                ["How all students in general pass or fail",
                 "What causes students to fail in real life",
                 "How accurate the model is"], rng)


def S5(e, rng):
    p, b = e["prob_pass"], e["base"]
    wrong = sorted({round(b, 2), round(min(max(p + 0.18, 0.02), 0.98), 2), round(min(max(p - 0.2, 0.02), 0.98), 2),
                    round(1 - p, 2)} - {round(p, 2)})
    return _mcq(f"The average prediction is {_pct(b)}. After adding all the bars, what is the chance of passing "
                "for this student?", _pct(p), [_pct(x) for x in wrong], rng)


def S6(e, rng):
    c = e["contrib"]
    ks = sorted(c, key=lambda k: -abs(c[k]))
    f3 = ks[0]
    f1, f2 = rng.sample(ks[1:], 2)
    combined, single = round(abs(c[f1] + c[f2]) * 1000), round(abs(c[f3]) * 1000)
    ans = "Yes" if combined > single else ("No" if combined < single else "They are equal")
    return _mcq(f"Taken together, did “{_lab(f1)}” and “{_lab(f2)}” change the prediction "
                f"MORE than “{_lab(f3)}” alone?", ans,
                ["Yes", "No", "They are equal", "Cannot be known from the explanation"], rng, keep_order=True)


def S7(e, rng):
    c = e["contrib"]
    top = max(c, key=lambda k: abs(c[k]))
    return _mcq(f"“{_lab(top)}” has a large bar. What does this show?",
                "That the model relied on it heavily for this prediction, not that it causes passing or failing",
                ["That it is the real cause of passing or failing for every student",
                 "That changing it will surely change the student’s real grade",
                 "That it is the most common factor among all students"], rng)


def S8(e, rng):
    c = e["contrib"]
    top = max(c, key=lambda k: abs(c[k]))
    ans = "Lower" if c[top] > 0 else "Higher"
    return _mcq(f"If “{_lab(top)}” had been at an average value instead, the chance of passing would most "
                "likely be…", ans, ["Higher", "Lower", "Exactly the same", "Exactly 50%"], rng, keep_order=True)


# ============================================================ LIME
def _conds(e):
    return e["conditions"]


def L1(e, rng):
    cs = _conds(e)
    top = max(cs, key=lambda x: abs(x["weight"]))
    others = [x["rule"] for x in cs if abs(abs(x["weight"]) - abs(top["weight"])) > LIME_TOL]
    shown = {x["feature"] for x in cs}
    others += [f"{F[k].label} > {fmt(k, F[k].default).split()[0].rstrip('%')}" for k in KEYS if k not in shown]
    return _mcq("Which condition has the BIGGEST influence on this prediction?", top["rule"], others, rng)


def L2(e, rng):
    cs = _conds(e)
    x = rng.choice([c for c in cs if abs(c["weight"]) >= LIME_TOL] or cs)
    ans = "Toward PASS" if x["weight"] > 0 else "Toward FAIL"
    return _mcq(f"Does the condition “{x['rule']}” push the prediction toward PASS or FAIL?", ans,
                ["Toward PASS", "Toward FAIL", "Neither", "It depends on other students"], rng, keep_order=True)


def L3(e, rng):
    n = sum(1 for x in _conds(e) if x["weight"] < 0)
    return _mcq(f"How many of the {len(_conds(e))} conditions shown push the prediction toward FAIL?",
                str(n), [str(k) for k in range(0, len(_conds(e)) + 1) if k != n], rng)


def L4(e, rng):
    return _mcq("How does LIME build this explanation?",
                "By fitting a simple model around this one student’s inputs",
                ["By listing the fixed rules the model always follows",
                 "By measuring what causes students to pass in real life",
                 "By averaging the results of all students in the school"], rng)


def L5(e, rng):
    return _mcq("Would this same LIME explanation apply to a student with very different inputs?",
                "No – it is only valid near this student’s inputs",
                ["Yes – it applies to every student",
                 "Yes – but only to students who passed",
                 "No – because LIME explanations are random"], rng)


def L6(e, rng):
    s = sum(round(x["weight"], 2) for x in _conds(e))
    ans = "The PASS side" if s > 0.004 else ("The FAIL side" if s < -0.004 else "They are exactly equal")
    return _mcq("Add up the weights of all the conditions. Which side is bigger?", ans,
                ["The PASS side", "The FAIL side", "They are exactly equal", "Weights cannot be added"], rng, keep_order=True)


def L7(e, rng):
    x = max(_conds(e), key=lambda c: abs(c["weight"]))
    pts = abs(x["weight"]) * 100
    direction = "raises" if x["weight"] > 0 else "lowers"
    return _mcq(f"The condition “{x['rule']}” has weight {x['weight']:+.2f}. What does this weight mean?",
                f"In LIME’s simple local model, meeting this condition {direction} the chance of passing by "
                f"about {pts:.0f} percentage points",
                [f"{pts:.0f}% of all students meet this condition",
                 f"The model is {100 - pts:.0f}% accurate",
                 "It is the probability that the condition is true"], rng)


def L8(e, rng):
    return _mcq("The conditions show ranges (for example “> 5”) instead of the exact values. Why?",
                "LIME groups each value into ranges to make the explanation easier to read",
                ["Because the exact values were lost",
                 "Because the model only accepts ranges as input",
                 "Because the ranges are the passing requirements of the school"], rng)


# ============================================================ Counterfactual
def _change_text(ch):
    return ", ".join(f"{_lab(c['feature'])} from {fmt(c['feature'], c['old'])} to {fmt(c['feature'], c['new'])}"
                     for c in ch)


def _fake_changes(e, rng, n):
    """Plausible wrong answers: changes to other factors, or the real changes in the wrong direction."""
    changed = {c["feature"] for c in e["changes"]}
    pool = [k for k in KEYS if F[k].actionable and k not in changed]
    out = []
    for k in rng.sample(pool, len(pool)):
        f, old = F[k], e["new_inputs"][k]
        delta = max(f.step, round((f.hi - f.lo) * 0.15 / f.step) * f.step)
        new = min(max(old + rng.choice([-1, 1]) * delta, f.lo), f.hi)
        if new == old:
            new = min(max(old - delta, f.lo), f.hi) if old + delta > f.hi else old + delta
        out.append(_change_text([{"feature": k, "old": old, "new": new}]))
    flipped = []
    for c in e["changes"]:
        f = F[c["feature"]]
        new = min(max(2 * c["old"] - c["new"], f.lo), f.hi)
        flipped.append({"feature": c["feature"], "old": c["old"], "new": new if new != c["old"] else c["new"]})
    if any(fc["new"] != c["new"] for fc, c in zip(flipped, e["changes"])):
        out.insert(1, _change_text(flipped))
    if len(e["changes"]) > 1:
        out.insert(0, _change_text(e["changes"][:1]))
        out.insert(2, _change_text(e["changes"][1:]))
    g = e["new_inputs"]["prev_grade"]
    out.append(_change_text([{"feature": "prev_grade", "old": g, "new": min(g + 10, 100) if g < 100 else 90}]))
    return out[:max(n, 4)]


def C1(e, rng):
    return _mcq("According to the counterfactual, which change would flip the prediction?",
                _change_text(e["changes"]), _fake_changes(e, rng, 3), rng)


def C2(e, rng):
    changed = [c["feature"] for c in e["changes"]]
    unchanged = [k for k in KEYS if k not in changed]
    ans = rng.choice(unchanged)
    dis = [_lab(k) for k in changed] + [_lab(k) for k in unchanged if k != ans]
    stem = ("Which factor is NOT changed in the counterfactual?")
    return _mcq(stem, _lab(ans), dis[:3] if len(changed) >= 1 else dis, rng)


def C3(e, rng):
    return _mcq("What would the model predict after the suggested changes?", e["new_label"],
                ["PASS", "FAIL", "The same as before", "The model cannot predict it"], rng, keep_order=True)


def C4(e, rng):
    c = rng.choice(e["changes"])
    ans = "It goes up" if c["new"] > c["old"] else "It goes down"
    return _mcq(f"In the counterfactual, what happens to “{_lab(c['feature'])}”?", ans,
                ["It goes up", "It goes down", "It stays the same", "It is removed from the model"], rng, keep_order=True)


def C5(e, rng):
    return _mcq("If the student really makes these changes, are they guaranteed to pass?",
                "No – it only shows what would change the MODEL’s prediction",
                ["Yes – the counterfactual guarantees the result",
                 "Yes – if they make the changes within one week",
                 "No – because counterfactuals are always wrong"], rng)


def C6(e, rng):
    if len(e["changes"]) > 1:
        a, b = e["changes"][:2]
        stem = (f"If the student made only the change to “{_lab(a['feature'])}” and not the one to "
                f"“{_lab(b['feature'])}”, would the prediction surely flip?")
    else:
        a = e["changes"][0]
        stem = (f"If the student changed “{_lab(a['feature'])}” by only HALF the suggested amount, "
                "would the prediction surely flip?")
    return _mcq(stem, "Not necessarily – the counterfactual is the smallest change that flips it",
                ["Yes, any change in that direction flips it", "Yes, but only after one term",
                 "No, the prediction can never flip"], rng)


def C7(e, rng):
    return _mcq("Why does the counterfactual never change the previous term grade?",
                "It cannot be changed now, so the search keeps it fixed",
                ["It has no effect on the prediction", "It is always 100%",
                 "The model does not use it"], rng)


def C8(e, rng):
    return _mcq("The counterfactual changes as few factors as possible. Why?",
                "To find the smallest, easiest change that flips the prediction",
                ["Because the model can only change one factor",
                 "To hide the other factors from the student",
                 "Because more changes would always make the student fail"], rng)


TEMPLATES = {
    "SHAP": [S1, S2, S3, S4, S5, S6, S7, S8],
    "LIME": [L1, L2, L3, L4, L5, L6, L7, L8],
    "CF": [C1, C2, C3, C4, C5, C6, C7, C8],
}
PREFIX = {"SHAP": "S", "LIME": "L", "CF": "C"}
# final check (post-test): 2 per type, separate item IDs so they are not part of the 24 calibrated items
FINAL = [("SHAP", S2), ("SHAP", S7), ("LIME", L2), ("LIME", L5), ("CF", C3), ("CF", C6)]


def item_meta(item_id: str) -> dict:
    kind = {"S": "SHAP", "L": "LIME", "C": "CF"}[item_id[0]]
    n = int(item_id[1:])
    return {"item_id": item_id, "skill": kind, "complexity": "L" if n <= 4 else "H"}


def build(kind: str, number: int, explanation: dict, seed: int) -> dict:
    """number 1..8 -> one generated question for item S1..S8 / L1..L8 / C1..C8."""
    q = TEMPLATES[kind][number - 1](explanation, random.Random(seed))
    q.update(item_meta(f"{PREFIX[kind]}{number}"))
    return q


def plan_for_user(seed: int) -> list[tuple[str, int, int]]:
    """12 rounds x 2 checks = 24 checks, 8 per type, every template exactly once.

    Each round uses one explanation type (rotating SHAP, LIME, CF) and asks one low- and one
    high-complexity template.  Returns [(kind, low_number, high_number)] per round.
    """
    rng = random.Random(seed)
    order = {k: (rng.sample([1, 2, 3, 4], 4), rng.sample([5, 6, 7, 8], 4)) for k in TYPES}
    start = rng.randrange(3)
    rounds = []
    for r in range(12):
        kind = TYPES[(start + r) % 3]
        lo, hi = order[kind]
        i = r // 3
        rounds.append((kind, lo[i], hi[i]))
    return rounds

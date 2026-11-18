"""Participant flow: register -> consent -> profile -> prior check -> tutorial -> 12 chat rounds -> final check -> results."""
from __future__ import annotations

import json
import os
import random
from functools import wraps

from flask import (Blueprint, abort, flash, redirect, render_template, request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

from . import get_predictor
from .display import inputs_text, view
from .models import Ask, Check, User, db, now
from .predictor import FEATURES, F, KEYS, parse_question
from .questions import FINAL, TYPE_NAMES, TYPES, build, item_meta, plan_for_user
from . import scoring

bp = Blueprint("study", __name__)
ROUNDS = 12
FINAL_INPUTS = {"study_hours": 5, "absences": 6, "sleep_hours": 6, "assignments_done": 70,
                "extracurricular_hours": 10, "prev_grade": 80}
EXAMPLES = [
    "Will I pass if I study 6 hours a week and have 3 absences?",
    "I sleep 5 hours a night and missed 8 classes. Will I pass?",
    "What if I study 2 hours per day but work 15 hours a week?",
    "My previous grade was 88% and I submitted 60% of my assignments. Pass or fail?",
]
_PK = None


def prior_items():
    global _PK
    if _PK is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "prior_knowledge.json")
        _PK = json.load(open(path, encoding="utf-8"))
    return _PK


# ------------------------------------------------------------------ helpers
def current_user():
    uid = session.get("uid")
    return db.session.get(User, uid) if uid else None


def stage_url(u: User):
    return {"consent": "study.consent", "profile": "study.profile", "prior": "study.prior",
            "tutorial": "study.tutorial", "chat": "study.chat", "final": "study.final",
            "done": "study.results"}[u.stage]


def login_required(stage=None):
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            u = current_user()
            if u is None:
                return redirect(url_for("study.login"))
            if stage and u.stage not in (stage if isinstance(stage, tuple) else (stage,)):
                return redirect(url_for(stage_url(u)))
            return fn(u, *a, **kw)
        return wrapper
    return deco


def make_checks(u: User, ask: Ask, specs, phase, first_order):
    for i, (kind, number, item_id) in enumerate(specs):
        e = ask.explanation[kind] if ask.xai_type == "ALL" else ask.explanation
        seed = (u.plan_seed or 0) * 1000 + (ask.round or 0) * 10 + i
        q = build(kind, number, e, seed)
        meta = item_meta(q["item_id"])
        db.session.add(Check(user_id=u.id, ask_id=ask.id, phase=phase, order=first_order + i,
                             item_id=item_id or meta["item_id"], skill=meta["skill"], complexity=meta["complexity"],
                             prompt=q["prompt"], options=q["options"], answer=q["answer"]))


# ------------------------------------------------------------------ public pages
@bp.route("/")
def index():
    u = current_user()
    if u:
        return redirect(url_for(stage_url(u)))
    return render_template("index.html")


@bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("username", "").strip().lower()
        pw = request.form.get("password", "")
        if not (3 <= len(name) <= 30) or not name.replace("_", "").replace(".", "").isalnum():
            flash("Username must be 3-30 letters or numbers (no spaces).")
        elif len(pw) < 6:
            flash("Password must have at least 6 characters.")
        elif pw != request.form.get("password2"):
            flash("The two passwords do not match.")
        elif User.query.filter_by(username=name).first():
            flash("That username is already taken.")
        else:
            u = User(username=name, pw_hash=generate_password_hash(pw), plan_seed=random.randrange(10 ** 6))
            db.session.add(u)
            db.session.commit()
            session.clear()
            session["uid"] = u.id
            return redirect(url_for("study.consent"))
    return render_template("register.html")


@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = User.query.filter_by(username=request.form.get("username", "").strip().lower()).first()
        if u and check_password_hash(u.pw_hash, request.form.get("password", "")):
            session.clear()
            session["uid"] = u.id
            return redirect(url_for(stage_url(u)))
        flash("Wrong username or password.")
    return render_template("login.html")


@bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("study.index"))


# ------------------------------------------------------------------ onboarding
@bp.route("/consent", methods=["GET", "POST"])
@login_required("consent")
def consent(u):
    if request.method == "POST":
        if request.form.get("agree") != "yes":
            flash("You need to agree before taking part. You can close this page if you do not want to join.")
        else:
            u.consent_at, u.stage = now(), "profile"
            db.session.commit()
            return redirect(url_for("study.profile"))
    return render_template("consent.html")


PROFILE_CHOICES = {
    "year_level": ["1", "2", "3", "4"],
    "prog_courses": ["1", "2", "3", "4"],
    "ai_courses": ["0", "1", "2", "3"],
}


@bp.route("/profile", methods=["GET", "POST"])
@login_required("profile")
def profile(u):
    if request.method == "POST":
        f = request.form
        if any(f.get(k) not in v for k, v in PROFILE_CHOICES.items()):
            flash("Please answer every question.")
        else:
            u.year_level, u.prog_courses, u.ai_courses = int(f["year_level"]), int(f["prog_courses"]), int(f["ai_courses"])
            u.stage = "prior"
            db.session.commit()
            return redirect(url_for("study.prior"))
    return render_template("profile.html", choices=PROFILE_CHOICES)


@bp.route("/prior", methods=["GET", "POST"])
@login_required("prior")
def prior(u):
    items = prior_items()
    if request.method == "POST":
        answers = {it["item_id"]: request.form.get(it["item_id"]) for it in items}
        if any(v is None for v in answers.values()):
            flash("Please answer all 10 questions (choose your best guess if unsure).")
            return render_template("prior.html", items=items, answers=answers)
        score = 0
        for k, it in enumerate(items, 1):
            ok = int(answers[it["item_id"]] == it["answer"])
            score += ok
            keys = [o["key"] for o in it["options"]]
            db.session.add(Check(user_id=u.id, phase="prior", order=k, item_id=it["item_id"], skill="PK",
                                 prompt=it["question"], options=[o["text"] for o in it["options"]],
                                 answer=keys.index(it["answer"]), chosen=keys.index(answers[it["item_id"]]),
                                 correct=ok, answered_at=now()))
        u.prior_score, u.stage = score, "tutorial"
        db.session.commit()
        return redirect(url_for("study.tutorial"))
    return render_template("prior.html", items=items, answers={})


@bp.route("/tutorial", methods=["GET", "POST"])
@login_required("tutorial")
def tutorial(u):
    if request.method == "POST":
        u.stage = "chat"
        db.session.commit()
        return redirect(url_for("study.chat"))
    p = get_predictor()
    sample = {"study_hours": 10, "absences": 5, "sleep_hours": 6, "assignments_done": 85,
              "extracurricular_hours": 4, "prev_grade": 84}
    pred = p.predict(sample)
    views = {k: view(k, p.explain(k, sample), sample) for k in TYPES}
    return render_template("tutorial.html", sample=inputs_text(sample), pred=pred, views=views)


# ------------------------------------------------------------------ chat rounds
def pending_ask(u):
    """The latest ask that still has unanswered checks (or None)."""
    c = Check.query.filter_by(user_id=u.id, correct=None).filter(Check.phase != "prior").order_by(Check.id).first()
    return db.session.get(Ask, c.ask_id) if c else None


@bp.route("/chat", methods=["GET", "POST"])
@login_required("chat")
def chat(u):
    a = pending_ask(u)
    if a:
        return redirect(url_for("study.answer", ask_id=a.id))
    kind = plan_for_user(u.plan_seed)[u.round][0]
    history = Ask.query.filter_by(user_id=u.id).filter(Ask.round <= ROUNDS).order_by(Ask.id.desc()).limit(3).all()
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if len(text) < 5:
            flash("Please type a question about a student's study habits.")
        else:
            inputs, found = parse_question(text)
            return render_template("confirm.html", text=text, inputs=inputs, found=found, features=FEATURES,
                                   kind=kind, round=u.round + 1, rounds=ROUNDS)
    return render_template("chat.html", round=u.round + 1, rounds=ROUNDS, kind=TYPE_NAMES[kind],
                           examples=EXAMPLES, history=history)


@bp.route("/ask", methods=["POST"])
@login_required("chat")
def ask(u):
    if pending_ask(u):
        return redirect(url_for("study.chat"))
    inputs = {}
    for f in FEATURES:
        try:
            v = float(request.form.get(f.key, f.default))
        except ValueError:
            v = f.default
        inputs[f.key] = float(min(max(round(v / f.step) * f.step, f.lo), f.hi))
    kind, lo, hi = plan_for_user(u.plan_seed)[u.round]
    p = get_predictor()
    e = p.explain(kind, inputs)
    if kind == "CF" and not e.get("found"):
        flash("For these values, no realistic change would flip the AI's prediction. "
              "Please ask about a different situation (for example, fewer absences or a higher previous grade).")
        return redirect(url_for("study.chat"))
    pred = p.predict(inputs)
    a = Ask(user_id=u.id, round=u.round + 1, text=request.form.get("text", ""), inputs=inputs,
            parsed=request.form.getlist("found"), label=pred["label"], prob_pass=pred["prob_pass"],
            xai_type=kind, explanation=e)
    db.session.add(a)
    db.session.flush()
    make_checks(u, a, [(kind, lo, None), (kind, hi, None)], "main", u.round * 2 + 1)
    db.session.commit()
    return redirect(url_for("study.answer", ask_id=a.id))


@bp.route("/answer/<int:ask_id>")
@login_required(("chat", "final", "done"))
def answer(u, ask_id):
    a = db.session.get(Ask, ask_id)
    if a is None or a.user_id != u.id:
        abort(404)
    checks = Check.query.filter_by(ask_id=a.id).order_by(Check.order).all()
    pending = next((c for c in checks if c.correct is None), None)
    fb = db.session.get(Check, request.args.get("fb", type=int) or -1)
    if fb is not None and fb.user_id != u.id:
        fb = None
    kinds = TYPES if a.xai_type == "ALL" else (a.xai_type,)
    views = {k: view(k, a.explanation[k] if a.xai_type == "ALL" else a.explanation, a.inputs) for k in kinds}
    n_done = sum(c.correct is not None for c in checks)
    return render_template("answer.html", a=a, views=views, kinds=kinds, names=TYPE_NAMES, pending=pending,
                           fb=fb, n_done=n_done, n_total=len(checks), inputs=inputs_text(a.inputs),
                           final=a.xai_type == "ALL", rounds=ROUNDS)


@bp.route("/check/<int:check_id>", methods=["POST"])
@login_required(("chat", "final"))
def check(u, check_id):
    c = db.session.get(Check, check_id)
    if c is None or c.user_id != u.id:
        abort(404)
    if c.correct is not None:
        return redirect(url_for("study.answer", ask_id=c.ask_id))
    try:
        chosen, conf, trust = (int(request.form[k]) for k in ("choice", "confidence", "trust"))
        assert 0 <= chosen < len(c.options) and 1 <= conf <= 5 and 1 <= trust <= 5
    except (KeyError, ValueError, AssertionError):
        flash("Please choose an answer and rate both your confidence and your trust.")
        return redirect(url_for("study.answer", ask_id=c.ask_id))
    c.chosen, c.confidence, c.trust = chosen, conf, trust
    c.correct, c.answered_at = int(chosen == c.answer), now()
    db.session.flush()
    remaining = Check.query.filter_by(ask_id=c.ask_id, correct=None).count()
    if remaining == 0:
        if c.phase == "main":
            u.round += 1
            if u.round >= ROUNDS:
                u.stage = "final"
        elif c.phase == "final":
            finals = Check.query.filter_by(user_id=u.id, phase="final").all()
            u.post_score = sum(x.correct for x in finals) / len(finals)
            u.stage, u.finished_at = "done", now()
    db.session.commit()
    return redirect(url_for("study.answer", ask_id=c.ask_id, fb=c.id))


@bp.route("/final")
@login_required("final")
def final(u):
    a = Ask.query.filter_by(user_id=u.id, round=99).first()
    if a is None:
        p = get_predictor()
        pred = p.predict(FINAL_INPUTS)
        a = Ask(user_id=u.id, round=99, text="Final check", inputs=FINAL_INPUTS, parsed=[], label=pred["label"],
                prob_pass=pred["prob_pass"], xai_type="ALL",
                explanation={k: p.explain(k, FINAL_INPUTS) for k in TYPES})
        db.session.add(a)
        db.session.flush()
        specs = []
        for i, (kind, fn) in enumerate(FINAL, 1):
            specs.append((kind, int(fn.__name__[1:]), f"F{i}"))
        make_checks(u, a, specs, "final", 1)
        db.session.commit()
    return redirect(url_for("study.answer", ask_id=a.id))


@bp.route("/results")
@login_required("done")
def results(u):
    checks = Check.query.filter_by(user_id=u.id).all()
    main = [c for c in checks if c.phase == "main"]
    by_type = {TYPE_NAMES[k]: (sum(c.correct for c in main if c.skill == k),
                               sum(1 for c in main if c.skill == k)) for k in TYPES}
    theta, se = scoring.ability(checks)
    from scipy.stats import norm
    mastery = {TYPE_NAMES[k]: v * 100 for k, v in scoring.mastery(checks).items()}
    rf = None
    try:
        rf = scoring.predicted_comprehension(u, checks)
    except Exception:          # model not calibrated yet or features missing
        rf = None
    return render_template("results.html", u=u, by_type=by_type, percentile=norm.cdf(theta) * 100,
                           mastery=mastery, rf=rf, proficient=(u.post_score or 0) >= 0.7)

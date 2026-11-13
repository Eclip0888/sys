"""Researcher / educator dashboard: progress, item statistics, data export and calibration."""
import io
from functools import wraps

import pandas as pd
from flask import (Blueprint, Response, current_app, flash, redirect, render_template, request, session, url_for)

from . import scoring
from .models import Check, User, db
from .questions import TYPE_NAMES, TYPES

bp = Blueprint("admin", __name__, url_prefix="/admin")


def admin_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if not session.get("admin"):
            return redirect(url_for("admin.login"))
        return fn(*a, **kw)
    return wrapper


@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("password") == current_app.config["ADMIN_PASSWORD"]:
            session["admin"] = True
            return redirect(url_for("admin.dashboard"))
        flash("Wrong admin password.")
    return render_template("admin_login.html")


@bp.route("/logout")
def logout():
    session.pop("admin", None)
    return redirect(url_for("study.index"))


@bp.route("/")
@admin_required
def dashboard():
    users = User.query.order_by(User.id).all()
    done = [u for u in users if u.stage == "done" and not u.excluded]
    stages = pd.Series([u.stage for u in users]).value_counts().to_dict() if users else {}
    _, resp = scoring.study_data(include_unfinished=True)
    stats = {"by_type": {}, "items": [], "conf": {}, "trust": {}}
    if not resp.empty:
        main = resp[resp["phase"] == "main"]
        for k in TYPES:
            m = main[main["skill"] == k]
            stats["by_type"][TYPE_NAMES[k]] = (m["correct"].mean() * 100 if len(m) else None,
                                               m["confidence"].mean() if len(m) else None,
                                               m["trust"].mean() if len(m) else None, len(m))
        it = main.groupby("item_id").agg(n=("correct", "size"), p=("correct", "mean"),
                                          conf=("confidence", "mean"), trust=("trust", "mean"))
        order = {f"{p}{n}": i for i, (p, n) in enumerate((p, n) for p in "SLC" for n in range(1, 9))}
        stats["items"] = sorted(it.reset_index().to_dict("records"), key=lambda r: order.get(r["item_id"], 99))
    minutes = [(u.finished_at - u.consent_at).total_seconds() / 60 for u in done if u.finished_at and u.consent_at]
    support = sorted((u for u in done if (u.post_score or 0) < 0.7), key=lambda u: u.post_score or 0)
    return render_template("admin.html", users=users, done=done, stages=stages, stats=stats,
                           mean_minutes=sum(minutes) / len(minutes) if minutes else None,
                           mean_post=sum(u.post_score for u in done) / len(done) * 100 if done else None,
                           support=support, calib=scoring.calibration_status(),
                           min_users=scoring.MIN_USERS_TO_CALIBRATE)


@bp.route("/calibrate", methods=["POST"])
@admin_required
def calibrate():
    try:
        r = scoring.calibrate()
        flash(r["message"])
    except Exception as e:          # show the reason instead of a server error
        flash(f"Calibration failed: {e}")
    return redirect(url_for("admin.dashboard"))


@bp.route("/exclude/<int:uid>", methods=["POST"])
@admin_required
def exclude(uid):
    u = db.session.get(User, uid)
    if u:
        u.excluded = not u.excluded
        db.session.commit()
    return redirect(url_for("admin.dashboard"))


def _csv(df, name):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={name}"})


@bp.route("/export/<what>.csv")
@admin_required
def export(what):
    users, resp = scoring.study_data(include_unfinished=request.args.get("all") == "1")
    if what == "responses":          # engine format: main checks only
        cols = ["user_id", "session", "order", "item_id", "skill", "complexity", "correct", "confidence", "trust",
                "answered_at"]
        df = resp[resp["phase"] == "main"][cols] if not resp.empty else pd.DataFrame(columns=cols)
        return _csv(df, "responses.csv")
    if what == "users":
        return _csv(scoring.users_frame(users), "users.csv")
    if what == "final":
        df = resp[resp["phase"] == "final"] if not resp.empty else pd.DataFrame()
        return _csv(df, "final_checks.csv")
    if what == "prior":
        df = resp[resp["phase"] == "prior"] if not resp.empty else pd.DataFrame()
        return _csv(df, "prior_knowledge.csv")
    if what == "questions":
        by_id = {u.id: u.code for u in users}
        rows = [{"user_id": by_id[c.user_id], "phase": c.phase, "order": c.order, "item_id": c.item_id,
                 "prompt": c.prompt, "options": " | ".join(c.options or []), "answer": c.answer, "chosen": c.chosen,
                 "correct": c.correct} for c in Check.query.filter(Check.user_id.in_(list(by_id) or [-1]))]
        return _csv(pd.DataFrame(rows), "generated_questions.csv")
    return "unknown export", 404

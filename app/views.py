"""User pages: account, chat, one-tap question and real-time evaluation."""
from __future__ import annotations

import time
from functools import wraps

import bleach
import markdown as md
from flask import Blueprint, abort, flash, redirect, render_template, request, session, url_for
from markupsafe import Markup
from werkzeug.security import check_password_hash, generate_password_hash

from . import engine
from .complexity import answer_features, question_length
from .gemini import GeminiError, get_client
from .models import Interaction, User, db, get_param, now
from .qgen import generate_question

bp = Blueprint("chat", __name__)
ALLOWED_TAGS = ["p", "br", "strong", "em", "ul", "ol", "li", "code", "pre", "h3", "h4", "blockquote", "table",
                "thead", "tbody", "tr", "th", "td"]


def render_md(text: str) -> Markup:
    html = md.markdown(text or "", extensions=["fenced_code", "tables", "sane_lists"])
    html = html.replace("<h1>", "<h3>").replace("</h1>", "</h3>").replace("<h2>", "<h3>").replace("</h2>", "</h3>")
    return Markup(bleach.clean(html, tags=ALLOWED_TAGS, strip=True))


def live_params() -> engine.Params:
    return engine.Params.from_dict(get_param("live_params")) if get_param("live_params") else engine.Params()


def current_user():
    uid = session.get("uid")
    return db.session.get(User, uid) if uid else None


def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        u = current_user()
        if u is None:
            return redirect(url_for("chat.login"))
        return fn(u, *a, **kw)
    return wrapper


@bp.route("/")
def index():
    if current_user():
        return redirect(url_for("chat.chat"))
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
            u = User(username=name, pw_hash=generate_password_hash(pw))
            db.session.add(u)
            db.session.commit()
            session.clear()
            session["uid"] = u.id
            return redirect(url_for("chat.chat"))
    return render_template("register.html")


@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = User.query.filter_by(username=request.form.get("username", "").strip().lower()).first()
        if u and check_password_hash(u.pw_hash, request.form.get("password", "")):
            session.clear()
            session["uid"] = u.id
            return redirect(url_for("chat.chat"))
        flash("Wrong username or password.")
    return render_template("login.html")


@bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("chat.index"))


def _history(u: User, before_id: int | None = None) -> list[dict]:
    q = Interaction.query.filter(Interaction.user_id == u.id, Interaction.correct.isnot(None))
    if before_id:
        q = q.filter(Interaction.id < before_id)
    return [{"x": i.correct, "b": i.b, "w": i.w} for i in q.order_by(Interaction.id)]


@bp.route("/chat")
@login_required
def chat(u):
    items = Interaction.query.filter_by(user_id=u.id).order_by(Interaction.id.desc()).limit(10).all()[::-1]
    pending = next((i for i in items if i.question_ok and i.correct is None and not i.skipped), None)
    if pending and pending.shown_at is None:
        pending.shown_at = now()
        db.session.commit()
    done = [i for i in Interaction.query.filter_by(user_id=u.id) if i.correct is not None]
    theta = done[-1].theta if done else None
    return render_template("chat.html", items=items, pending=pending, render_md=render_md, theta=theta,
                           n_done=len(done), understood=sum(1 for i in done if i.label == "Understood"))


@bp.route("/ask", methods=["POST"])
@login_required
def ask(u):
    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return redirect(url_for("chat.chat"))
    if len(prompt) > 1500:
        flash("Please keep your question under 1,500 characters.")
        return redirect(url_for("chat.chat"))
    pending = Interaction.query.filter_by(user_id=u.id, question_ok=True, correct=None, skipped=False).first()
    if pending:
        flash("Please answer or skip the question about the last answer first.")
        return redirect(url_for("chat.chat"))
    client = get_client()
    it = Interaction(user_id=u.id, prompt=prompt)
    try:
        t0 = time.monotonic()
        it.answer = client.answer(prompt)
        it.answer_seconds = round(time.monotonic() - t0, 2)
    except GeminiError as e:
        flash(f"The AI could not answer right now ({e}). Please try again in a minute.")
        return redirect(url_for("chat.chat"))
    feats = answer_features(it.answer)
    feats["model"] = getattr(client, "last_model", "?")
    try:
        qr = generate_question(client, prompt, it.answer)
    except GeminiError as e:
        qr = None
        it.qgen_error = str(e)
    if qr and qr.ok:
        it.question_ok, it.question, it.options, it.key, it.level = True, qr.question, qr.options, qr.key, qr.level
        it.question_words = question_length(qr.question, qr.options)
        feats["level"] = qr.level
    elif qr:
        it.question_ok, it.qgen_error = False, qr.error
    it.attempts = qr.attempts if qr else 0
    it.features = feats
    db.session.add(it)
    db.session.commit()
    return redirect(url_for("chat.chat") + f"#i{it.id}")


@bp.route("/respond/<int:iid>", methods=["POST"])
@login_required
def respond(u, iid):
    it = db.session.get(Interaction, iid)
    if it is None or it.user_id != u.id:
        abort(404)
    if not it.question_ok or it.correct is not None or it.skipped:
        return redirect(url_for("chat.chat"))
    if request.form.get("skip"):
        it.skipped, it.answered_at = True, now()
        db.session.commit()
        return redirect(url_for("chat.chat"))
    try:
        chosen = int(request.form["choice"])
        assert 0 <= chosen < len(it.options)
    except (KeyError, ValueError, AssertionError):
        abort(400)
    # response time: measured in the browser (question shown -> tap); server time as a fallback
    try:
        seconds = float(request.form.get("elapsed_ms", "")) / 1000.0
        assert 0 < seconds < 3600
    except (ValueError, AssertionError):
        seconds = (now() - (it.shown_at or it.created_at)).total_seconds()
    t0 = time.perf_counter()
    ev = engine.evaluate(_history(u, before_id=it.id), it.features, chosen == it.key, seconds,
                         it.question_words or 30, live_params())
    it.engine_ms = round((time.perf_counter() - t0) * 1000, 3)
    it.chosen, it.correct, it.seconds, it.answered_at = chosen, int(chosen == it.key), round(seconds, 2), now()
    it.b, it.w, it.theta, it.score, it.label, it.reason = ev.b, ev.w, ev.theta, ev.score, ev.label, ev.reason
    db.session.commit()
    return redirect(url_for("chat.chat") + f"#i{it.id}")

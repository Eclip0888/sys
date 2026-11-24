"""Researcher / admin pages: evaluation log, prompt-set run, simulation and parameters."""
import io
from functools import wraps

import markdown as md
from flask import (Blueprint, Response, current_app, flash, redirect, render_template, request, session, url_for)
from markupsafe import Markup

from . import engine, jobs, reports
from .gemini import get_client
from .models import Interaction, PromptRun, User, db, get_param, set_param

bp = Blueprint("admin", __name__, url_prefix="/admin")


def admin_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if not session.get("admin"):
            return redirect(url_for("admin.login"))
        return fn(*a, **kw)
    return wrapper


def _csv(df, name):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response("﻿" + buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={name}"})


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
    return redirect(url_for("chat.index"))


@bp.route("/")
@admin_required
def dashboard():
    its = Interaction.query.all()
    answered = [i for i in its if i.correct is not None]
    labels = {k: sum(1 for i in answered if i.label == k) for k in ("Understood", "Partly understood", "Not understood")}
    ms = [i.engine_ms for i in answered if i.engine_ms is not None]
    p = engine.Params.from_dict(get_param("live_params")) if get_param("live_params") else engine.Params()
    client = get_client()
    return render_template("admin.html", n_users=User.query.count(), n_inter=len(its), n_answered=len(answered),
                           n_qfail=sum(1 for i in its if not i.question_ok), labels=labels,
                           mean_ms=sum(ms) / len(ms) if ms else None, params=p, model=getattr(client, "model", "?"),
                           offline=client.fake, recent=sorted(its, key=lambda i: -i.id)[:30])


@bp.route("/export/interactions.csv")
@admin_required
def export_interactions():
    return _csv(reports.interactions_frame(), "interactions.csv")


# ------------------------------------------------------------------ prompt set
@bp.route("/prompts", methods=["GET", "POST"])
@admin_required
def prompts():
    batch = request.values.get("batch", "main").strip() or "main"
    if request.method == "POST":
        f = request.files.get("csvfile")
        if f and f.filename:
            text = f.read().decode("utf-8", errors="replace")
        else:
            import os
            text = open(os.path.join(current_app.root_path, "..", "data", "prompts_200.csv"), encoding="utf-8").read()
        items = jobs.read_prompt_csv(text)
        if not items:
            flash("No prompts found. The CSV needs a column named Prompt.")
        elif jobs.start_prompt_set(current_app._get_current_object(), items, batch):
            flash(f"Started: {len(items)} prompts in batch '{batch}'. Prompts already answered in this batch are skipped.")
        else:
            flash("A prompt-set run is already in progress.")
        return redirect(url_for("admin.prompts", batch=batch))
    runs = reports.prompt_runs(batch)
    batches = sorted({b for (b,) in db.session.query(PromptRun.batch).distinct()})
    return render_template("admin_prompts.html", batch=batch, batches=batches, runs=runs,
                           t2=reports.table2(runs), t3=reports.table3(runs), status=get_param("job_prompts", {}),
                           running=jobs.running("prompts"), offline=get_client().fake)


@bp.route("/prompts/export/<what>.csv")
@admin_required
def export_prompts(what):
    runs = reports.prompt_runs(request.args.get("batch", "main"))
    if what == "results":
        return _csv(reports.runs_frame(runs), "prompt_set_results.csv")
    if what == "review":
        return _csv(reports.review_frame(runs), "question_review_50.csv")
    return "unknown export", 404


@bp.route("/prompts/delete", methods=["POST"])
@admin_required
def delete_batch():
    batch = request.form.get("batch", "")
    if jobs.running("prompts"):
        flash("Wait for the current run to finish first.")
    elif request.form.get("confirm") == batch and batch:
        n = PromptRun.query.filter_by(batch=batch).delete()
        db.session.commit()
        flash(f"Deleted {n} results of batch '{batch}'.")
    else:
        flash("Type the batch name to confirm the delete.")
    return redirect(url_for("admin.prompts"))


# ------------------------------------------------------------------ simulation
@bp.route("/simulation", methods=["GET", "POST"])
@admin_required
def simulation():
    if request.method == "POST":
        try:
            s = {"replications": max(2, min(int(request.form.get("replications", 100)), 500)),
                 "users": max(20, min(int(request.form.get("users", 200)), 5000)),
                 "calibration_sets": max(1, min(int(request.form.get("calibration_sets", 20)), 100)),
                 "batch": request.form.get("batch") or None}
        except ValueError:
            flash("Please enter whole numbers.")
            return redirect(url_for("admin.simulation"))
        if jobs.start_simulation(current_app._get_current_object(), s):
            flash("Simulation started. This page refreshes while it runs.")
        else:
            flash("A simulation is already running.")
        return redirect(url_for("admin.simulation"))
    rep = get_param("sim_report")
    html = Markup(md.markdown(rep["markdown"], extensions=["tables"])) if rep else None
    batches = sorted({b for (b,) in db.session.query(PromptRun.batch).distinct()})
    return render_template("admin_sim.html", rep=rep, html=html, status=get_param("job_sim", {}),
                           running=jobs.running("sim"), batches=batches)


@bp.route("/simulation/export/<what>.csv")
@admin_required
def export_sim(what):
    rep = get_param("sim_report")
    key = {"results": "results_csv", "means": "means_csv", "calibration": "calibration_csv"}.get(what)
    if not rep or not key:
        return "no simulation results yet", 404
    return Response("﻿" + rep[key], mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=simulation_{what}.csv"})


@bp.route("/params", methods=["POST"])
@admin_required
def params():
    action = request.form.get("action")
    if action == "use_calibrated":
        rep = get_param("sim_report")
        if rep:
            set_param("live_params", rep["calibrated"])
            flash("The chatbot now uses the calibrated parameters from the last simulation.")
    elif action == "reset":
        set_param("live_params", None)
        flash("The chatbot now uses the default parameters.")
    return redirect(url_for("admin.dashboard"))

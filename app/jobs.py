"""Background jobs: the prompt-set run (Data Generation) and the simulation study.

Each job runs in a thread inside the web process and saves its progress in the Param table, so the admin
page can show it. Long runs can also be done from the command line (scripts/).
"""
from __future__ import annotations

import csv
import io
import threading
import time
import traceback

import pandas as pd

from .complexity import answer_features, question_length
from .gemini import GeminiError, get_client
from .models import PromptRun, db, get_param, set_param
from .qgen import generate_question

_threads: dict[str, threading.Thread] = {}


def running(name: str) -> bool:
    t = _threads.get(name)
    return t is not None and t.is_alive()


def _status(name, **kw):
    s = get_param(f"job_{name}", {}) or {}
    s.update(kw)
    s["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    set_param(f"job_{name}", s)


# ------------------------------------------------------------------ prompt set
def read_prompt_csv(text: str) -> list[dict]:
    """Accepts the 200-prompt sheet exported as CSV (No., Prompt, Domain, Target difficulty, ...)."""
    rows = []
    for i, r in enumerate(csv.DictReader(io.StringIO(text.lstrip("﻿"))), 1):
        r = {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
        prompt = r.get("prompt", "")
        if not prompt:
            continue
        num = r.get("no.") or r.get("no") or r.get("number") or str(i)
        try:
            num = int(float(num))
        except ValueError:
            num = i
        rows.append({"number": num, "prompt": prompt, "domain": r.get("domain", ""),
                     "target": r.get("target difficulty", r.get("difficulty", ""))})
    return rows


def process_prompt(client, batch: str, item: dict) -> PromptRun:
    pr = PromptRun(batch=batch, number=item["number"], domain=item["domain"], target=item["target"],
                   prompt=item["prompt"])
    try:
        t0 = time.monotonic()
        pr.answer = client.answer(item["prompt"])
        pr.answer_seconds = round(time.monotonic() - t0, 2)
        feats = answer_features(pr.answer)
        q = generate_question(client, item["prompt"], pr.answer)
        pr.attempts, pr.valid_outputs, pr.first_try, pr.question_seconds = q.attempts, q.valid_outputs, q.first_try, q.seconds
        pr.question_ok = q.ok
        if q.ok:
            pr.question, pr.options, pr.key, pr.level = q.question, q.options, q.key, q.level
            pr.question_words = question_length(q.question, q.options)
            feats["level"] = q.level
        else:
            pr.error = q.error
        pr.features = feats
    except GeminiError as e:
        pr.error = str(e)
        pr.question_ok = False
    return pr


def run_prompt_set(app, items: list[dict], batch: str):
    with app.app_context():
        client = get_client()
        done_numbers = {n for (n,) in db.session.query(PromptRun.number).filter(PromptRun.batch == batch,
                                                                                    PromptRun.answer.isnot(None))}
        todo = [it for it in items if it["number"] not in done_numbers]
        _status("prompts", state="running", batch=batch, done=len(items) - len(todo), total=len(items),
                message=f"Model: {getattr(client, 'model', '?')}", error="")
        failures = 0
        for it in todo:
            # remove an earlier failed try of this prompt, then process it again
            PromptRun.query.filter_by(batch=batch, number=it["number"]).delete()
            pr = process_prompt(client, batch, it)
            db.session.add(pr)
            db.session.commit()
            failures = failures + 1 if pr.answer is None else 0
            _status("prompts", done=get_param("job_prompts")["done"] + 1, last=f"#{it['number']}")
            if failures >= 5:
                _status("prompts", state="stopped",
                        error=f"Stopped after 5 failed calls in a row (last: {pr.error}). Your Gemini quota may be "
                              "used up for today; press Start again later to continue where it stopped.")
                return
        _status("prompts", state="finished")


def start_prompt_set(app, items, batch):
    if running("prompts"):
        return False
    t = threading.Thread(target=_safe, args=("prompts", run_prompt_set, app, items, batch), daemon=True)
    _threads["prompts"] = t
    t.start()
    return True


# ------------------------------------------------------------------ simulation
def pool_from_db(batch: str | None = None) -> pd.DataFrame | None:
    q = PromptRun.query.filter(PromptRun.question_ok.is_(True))
    if batch:
        q = q.filter(PromptRun.batch == batch)
    rows = [{"fkgl": r.features["fkgl"], "len": r.features["len"], "tech": r.features["tech"],
             "level": r.features.get("level", 2), "qwords": r.question_words or 40} for r in q]
    return pd.DataFrame(rows) if len(rows) >= 30 else None


def run_simulation(app, settings: dict):
    from sim.simulation import Config, calibrated_params, report_markdown, run_study, synthetic_pool
    with app.app_context():
        pool = pool_from_db(settings.get("batch"))
        source = f"{len(pool)} Gemini answers from the prompt set" if pool is not None else "synthetic (no prompt-set answers yet)"
        if pool is None:
            pool = synthetic_pool()
        cfg = Config(replications=settings["replications"], users=settings["users"],
                     calibration_sets=settings["calibration_sets"], pool_source=source)
        _status("sim", state="running", done=0, total=1, message=source, error="")
        rep = run_study(pool, cfg, progress=lambda d, t: _status("sim", done=d, total=t))
        set_param("sim_report", {"markdown": report_markdown(rep), "decisions": rep["decisions"],
                                 "runtime": rep["runtime"], "config": rep["config"],
                                 "calibrated": calibrated_params(rep),
                                 "results_csv": rep["frames"]["results"].to_csv(index=False),
                                 "means_csv": rep["mean"].to_csv(index=False),
                                 "calibration_csv": rep["frames"]["calibration"].to_csv(index=False)})
        _status("sim", state="finished")


def start_simulation(app, settings):
    if running("sim"):
        return False
    t = threading.Thread(target=_safe, args=("sim", run_simulation, app, settings), daemon=True)
    _threads["sim"] = t
    t.start()
    return True


def _safe(name, fn, app, *args):
    try:
        fn(app, *args)
    except Exception as e:                       # report the error on the admin page
        with app.app_context():
            db.session.rollback()
            _status(name, state="error", error=f"{e}", trace=traceback.format_exc()[-1500:])

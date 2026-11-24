"""Summaries for Chapter 4 (Tables 2 and 3) and CSV exports."""
from __future__ import annotations

import random

import numpy as np
import pandas as pd

from .models import Interaction, PromptRun


def prompt_runs(batch=None) -> list[PromptRun]:
    q = PromptRun.query
    if batch:
        q = q.filter_by(batch=batch)
    return q.order_by(PromptRun.number).all()


def table2(runs) -> list[dict]:
    """Complexity of the Gemini answers."""
    answered = [r for r in runs if r.features]
    rows = []
    for key, label, fn in [("fkgl", "Flesch–Kincaid Grade Level", float), ("words", "Word count", float),
                           ("tech", "Technical-term ratio", float), ("level", "Question cognitive level (1–3)", float)]:
        v = np.array([fn(r.features[key]) for r in answered if r.features.get(key) is not None])
        rows.append({"feature": label, "n": len(v), "mean": v.mean() if len(v) else None,
                     "sd": v.std(ddof=1) if len(v) > 1 else None, "min": v.min() if len(v) else None,
                     "max": v.max() if len(v) else None})
    return rows


def table3(runs) -> dict:
    answered = [r for r in runs if r.answer]
    n = len(answered)
    attempts = sum(r.attempts or 0 for r in answered)
    valid = sum(r.valid_outputs or 0 for r in answered)
    secs = [(r.answer_seconds or 0) + (r.question_seconds or 0) for r in answered]
    pct = lambda k: 100.0 * k / n if n else None
    return {"prompts": len(runs), "answered": n, "failed_calls": len(runs) - n,
            "valid_output_pct": 100.0 * valid / attempts if attempts else None,
            "first_try_pct": pct(sum(1 for r in answered if r.first_try)),
            "accepted_pct": pct(sum(1 for r in answered if r.question_ok)),
            "mean_seconds": float(np.mean(secs)) if secs else None,
            "by_domain": pd.DataFrame([{"domain": r.domain, "fkgl": r.features["fkgl"], "words": r.features["words"],
                                        "tech": r.features["tech"]} for r in answered if r.features])
                           .groupby("domain").mean().round(2).reset_index().to_dict("records") if n else []}


def runs_frame(runs) -> pd.DataFrame:
    return pd.DataFrame([{
        "no": r.number, "domain": r.domain, "target_difficulty": r.target, "prompt": r.prompt, "answer": r.answer,
        "fkgl": (r.features or {}).get("fkgl"), "words": (r.features or {}).get("words"),
        "tech": (r.features or {}).get("tech"), "level": r.level, "question_ok": r.question_ok,
        "question": r.question, "option_a": (r.options or [None] * 4)[0], "option_b": (r.options or [None] * 4)[1],
        "option_c": (r.options or [None] * 4)[2], "option_d": (r.options or [None] * 4)[3],
        "key": "ABCD"[r.key] if r.key is not None and r.key >= 0 else None, "attempts": r.attempts,
        "valid_outputs": r.valid_outputs, "first_try": r.first_try, "question_words": r.question_words,
        "answer_seconds": r.answer_seconds, "question_seconds": r.question_seconds, "error": r.error,
    } for r in runs])


def review_frame(runs, n=50, seed=2026) -> pd.DataFrame:
    """Random sample of accepted questions with the Chapter 3 review checklist."""
    ok = [r for r in runs if r.question_ok]
    sample = sorted(random.Random(seed).sample(ok, min(n, len(ok))), key=lambda r: r.number)
    df = runs_frame(sample)[["no", "domain", "prompt", "answer", "question", "option_a", "option_b", "option_c",
                             "option_d", "key", "level"]]
    for col in ("key_correct (Y/N)", "only_one_correct (Y/N)", "answerable_from_answer (Y/N)",
                "clear_wording (Y/N)", "reviewer", "notes"):
        df[col] = ""
    return df


def interactions_frame() -> pd.DataFrame:
    return pd.DataFrame([{
        "id": i.id, "user": f"U{i.user_id:04d}", "created_at": i.created_at, "prompt": i.prompt,
        "answer_words": (i.features or {}).get("words"), "fkgl": (i.features or {}).get("fkgl"),
        "tech": (i.features or {}).get("tech"), "level": i.level, "question_ok": i.question_ok, "question": i.question,
        "key": i.key, "chosen": i.chosen, "correct": i.correct, "skipped": i.skipped, "seconds": i.seconds,
        "b": i.b, "w": i.w, "theta": i.theta, "score": i.score, "label": i.label, "engine_ms": i.engine_ms,
        "answer_seconds": i.answer_seconds,
    } for i in Interaction.query.order_by(Interaction.id)])

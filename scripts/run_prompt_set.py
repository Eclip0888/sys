"""Run the prompt set through Gemini from your computer (Chapter 3, Data Generation).

    set GEMINI_API_KEY=...            (Windows)   /   export GEMINI_API_KEY=...   (Mac/Linux)
    python scripts/run_prompt_set.py --csv data/prompts_200.csv --batch main

Results go to the same database as the website (instance/chatbot.db, or DATABASE_URL if set), and the
CSV files for Chapter 4 are written to results/. Stopped runs continue where they stopped.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import create_app, jobs, reports  # noqa: E402
from app.gemini import get_client  # noqa: E402
from app.models import get_param  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/prompts_200.csv")
ap.add_argument("--batch", default="main")
ap.add_argument("--out", default="results")
a = ap.parse_args()
app = create_app()
items = jobs.read_prompt_csv(open(a.csv, encoding="utf-8").read())
with app.app_context():
    print(f"{len(items)} prompts · model {getattr(get_client(), 'model', '?')}"
          + (" · OFFLINE (no GEMINI_API_KEY)" if get_client().fake else ""))
jobs.run_prompt_set(app, items, a.batch)
with app.app_context():
    s = get_param("job_prompts")
    print(f"status: {s['state']} {s.get('done')}/{s.get('total')} {s.get('error', '')}")
    runs = reports.prompt_runs(a.batch)
    os.makedirs(a.out, exist_ok=True)
    reports.runs_frame(runs).to_csv(os.path.join(a.out, "prompt_set_results.csv"), index=False)
    reports.review_frame(runs).to_csv(os.path.join(a.out, "question_review_50.csv"), index=False)
    t3 = reports.table3(runs)
    print({k: v for k, v in t3.items() if k != "by_domain"})
    print(f"saved {a.out}/prompt_set_results.csv and {a.out}/question_review_50.csv")

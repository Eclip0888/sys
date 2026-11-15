# XAI-Check

A web-based AI Assistant that checks whether users understand its explanations. It was built for the thesis
*Augmentation of Item Response Theory and Bayesian Knowledge Tracing with Explainable Random Forest in an AI
Assistant for Automated Evaluation of User Comprehension of AI Outputs* (BS Computer Science, Pamantasan ng
Lungsod ng Muntinlupa).

## What a participant does (about 40 minutes)

1. Creates an account, agrees to the consent form and fills in a short profile.
2. Answers a 10-item warm-up on AI basics (the prior knowledge check).
3. Reads a short tutorial on SHAP, LIME and counterfactual explanations.
4. Asks the AI 12 questions in their own words, for example "Will I pass if I study 6 hours a week and
   have 3 absences?". For each question the AI:
   - reads the numbers in the question, and the user confirms or edits them;
   - predicts PASS or FAIL and explains the prediction with SHAP, LIME or a counterfactual (rotating);
   - automatically generates two comprehension questions about that explanation, one of low and one of
     high complexity, from 24 calibrated templates;
   - collects the answer, the user's confidence (1–5) and their trust in the explanation (1–5), and
     shows immediate feedback.
5. Takes a final check: a new student explained in all three ways, with 6 questions (the post-test).
6. Sees their results: final check score, mastery per explanation type and, after calibration, the
   Random Forest estimate.

Every user answers each of the 24 templates exactly once, 8 per explanation type, so the templates can be
calibrated as IRT items.

## Project layout

```
app/
  predictor.py   AI model (Random Forest, PASS/FAIL), question parser, SHAP / LIME / counterfactual
  questions.py   24 question templates (S1-S8, L1-L8, C1-C8) and the final check
  scoring.py     live scoring (augmented IRT ability, augmented BKT mastery, Random Forest) and calibration
  views.py       participant flow
  admin.py       researcher dashboard, CSV export, calibration
  templates/     HTML pages (mobile-first)
engine/          evaluation engine: augmented IRT, augmented BKT, Random Forest + SHAP, statistics (H1-H5)
scripts/
  run_evaluation.py       runs H1-H5, the ablation study and the baselines on the exported CSVs
  parameter_recovery.py   recovery of λ and P(T | τ) on 50 synthetic datasets (Table 15)
  performance_tests.py    time, memory, live scoring, robustness and scalability (Table 3)
  assistments_benchmark.py  standard IRT and BKT on the ASSISTments 2009-2010 dataset
  simulate_users.py       dry run: simulated participants go through the whole website
data/prior_knowledge.json   the 10 warm-up items
tests/           app and engine tests (pytest)
```

## Run it on your computer

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python wsgi.py                     # open http://127.0.0.1:5000
```

The researcher dashboard is at `/admin`. The default password is `admin`, so set `ADMIN_PASSWORD` before
going online.

Run the tests with `pip install pytest` and then `pytest -q`.

## Put it online (Render + Neon, free tier)

**1. Create the database (Neon).**
Sign in at https://neon.tech with GitHub → **Create project** (region: *AWS Asia Pacific (Singapore)*,
the closest to the Philippines) → on the dashboard press **Connect** and copy the connection string. It
looks like `postgresql://user:password@ep-xxxx.ap-southeast-1.aws.neon.tech/neondb?sslmode=require`.
Keep it private: anyone with it can read the data.

**2. Create the web service (Render).**
Sign in at https://render.com with GitHub → **New → Blueprint** → select this repository. `render.yaml`
fills in the build and start commands. When asked for the environment variables, enter:

| Variable | Value |
|---|---|
| `DATABASE_URL` | the Neon connection string from step 1 |
| `ADMIN_PASSWORD` | your own strong password for `/admin` (not `admin`) |
| `SECRET_KEY` | generated automatically |

The first build takes about 5–10 minutes.

**3. Check it.**
Open the `.onrender.com` URL on a phone and do one full sitting with a test account. Then open
`/admin`, log in, and confirm the test user appears. In Neon's **Tables** view you should see the same
user, which proves the data is stored in PostgreSQL and not in Render's temporary disk.
Exclude or delete the test user before real data collection.

**Notes.**
- Without `DATABASE_URL` the app falls back to SQLite, which **Render deletes on every restart**. Never
  collect real data that way.
- The free Render tier sleeps after 15 minutes without visitors; the first visit after that takes about a
  minute. Open the site a few minutes before each class session.
- `postgres://` and `postgresql://` URLs both work; the app selects the installed psycopg2 driver.

## During the study

| When | What to do in `/admin` |
|---|---|
| After the pilot (5–10 users) | Check the item table: templates that almost everyone or almost no one answers correctly may need rewording. Then delete the pilot data (use a fresh database). |
| After 30 or more users | Press **Calibrate**. The augmented IRT, BKT and Random Forest are fitted on the real answers, and new users are scored with them. |
| At the end | Download `responses.csv` and `users.csv`, then run the evaluation below. |

```bash
python scripts/run_evaluation.py --responses responses.csv --users users.csv
```

This writes `results/evaluation_report.md` (H1–H5), `table16_ablation.csv` and `table18_baselines.csv`
for Chapter 4.

### Algorithm evaluation that does not need participants

These can be run now, before data collection:

```bash
python scripts/parameter_recovery.py      # Table 15: 50 synthetic datasets, bias / RMSE / 95% CI coverage
python scripts/performance_tests.py       # Table 3: time, memory, live scoring, robustness, scalability
python scripts/assistments_benchmark.py --data data/assistments_2009.csv   # ASSISTments benchmark
```

The ASSISTments 2009–2010 skill-builder file (about 80 MB) is not stored in the repository. Download it with

```bash
curl -L -o data/assistments_2009.csv https://raw.githubusercontent.com/CAHLR/pyBKT-examples/master/data/as.csv
```

After data collection, run the performance tests on the real export as well:
`python scripts/performance_tests.py --responses responses.csv --users users.csv`. Runtimes depend on the
computer, so report the machine used (the script prints it).

## Notes for the thesis

- **Prediction model:** a Random Forest trained on a synthetic academic-performance dataset generated with a
  documented rule (`make_dataset` in `app/predictor.py`). It can be swapped for a real dataset without
  changing anything else.
- **Counterfactuals:** found by searching the smallest change to the actionable factors (up to three) that
  flips the prediction, DiCE-style. The previous grade is kept fixed. For fewer than 2% of extreme inputs no
  realistic change flips the prediction; the app then asks the user for a different scenario.
- **Data:** stored under codes (U0001, …), never names, in line with the Data Privacy Act of 2012.

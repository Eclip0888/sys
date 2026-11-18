"""Dry run: simulated participants go through the real website (Flask test client).

Each simulated user has an ability; better users pick the correct option more often, and
their confidence and trust ratings follow their correctness loosely.  Use it to check the
whole pipeline (sitting -> export -> calibration -> evaluation) before the pilot.

    python scripts/simulate_users.py --users 35 --db /tmp/sim.db
"""
import argparse
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import create_app            # noqa: E402
from app.models import Check, db      # noqa: E402
from app.views import prior_items     # noqa: E402

TEXTS = ["Will I pass if I study {s} hours a week and have {a} absences?",
         "I sleep {sl} hours a night, missed {a} classes and study {s} hours weekly. Will I pass?",
         "What if I study {s} hours a week, work {w} hours a week, and my previous grade was {g}%?"]


def sitting(c, app, k, rng):
    skill = rng.gauss(0, 1)
    p_ok = lambda: 1 / (1 + 2.718 ** -(0.3 + 1.1 * skill))
    c.post("/register", data={"username": f"sim{k:03d}", "password": "secret1", "password2": "secret1"})
    c.post("/consent", data={"agree": "yes"})
    c.post("/profile", data={"year_level": str(rng.randint(2, 4)), "prog_courses": str(rng.randint(1, 4)),
                             "ai_courses": str(rng.randint(0, 3))})
    c.post("/prior", data={it["item_id"]: (it["answer"] if rng.random() < p_ok() else "A") for it in prior_items()})
    c.post("/tutorial", data={})

    def answer_checks(loc, n):
        for _ in range(n):
            page = c.get(loc).data.decode()
            cid = int(re.search(r"/check/(\d+)", page).group(1))
            with app.app_context():
                chk = db.session.get(Check, cid)
                right, nopt = chk.answer, len(chk.options)
            ok = rng.random() < p_ok()
            choice = right if ok else rng.choice([i for i in range(nopt) if i != right])
            conf = max(1, min(5, round(3 + (0.8 if ok else -0.5) + rng.gauss(0, 1))))
            trust = max(1, min(5, round(3 + 0.4 * skill + rng.gauss(0, 1))))
            c.post(f"/check/{cid}", data={"choice": choice, "confidence": conf, "trust": trust})

    for _ in range(12):
        while True:
            vals = dict(s=rng.randint(2, 20), a=rng.randint(0, 9), sl=rng.choice([5, 6, 7, 8]),
                        w=rng.randint(0, 15), g=rng.randint(72, 95))
            text = rng.choice(TEXTS).format(**vals)
            page = c.post("/chat", data={"text": text}).data.decode()
            form = dict(re.findall(r'name="(\w+)" value="([\d.]+)" min', page))
            r = c.post("/ask", data={"text": text, **form})
            if "/answer/" in (r.location or ""):
                break
        answer_checks(r.location, 2)
    r = c.get("/final")
    answer_checks(r.location, 6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=35)
    ap.add_argument("--db", default="/tmp/xai_sim.db")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if os.path.exists(a.db):
        os.remove(a.db)
    app = create_app({"SQLALCHEMY_DATABASE_URI": f"sqlite:///{a.db}", "TESTING_NO_CSRF": True})
    rng = random.Random(a.seed)
    for k in range(a.users):
        with app.test_client() as c:
            sitting(c, app, k, rng)
        print(f"user {k + 1}/{a.users} done", flush=True)
    print(f"database: {a.db}")

"""End-to-end: register, ask, tap, see the evaluation; admin pages and exports."""
import re

import pytest

from app import create_app, database_url
from app.models import Interaction, PromptRun, db


@pytest.fixture()
def client(tmp_path):
    app = create_app({"SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path}/t.db", "TESTING": True,
                      "TESTING_NO_CSRF": True, "ADMIN_PASSWORD": "pw"})
    with app.test_client() as c:
        c.app = app
        yield c


def register(c, name="tester"):
    return c.post("/register", data={"username": name, "password": "secret1", "password2": "secret1"})


def test_database_url():
    assert database_url("postgres://u:p@h/db") == "postgresql+psycopg2://u:p@h/db"
    assert database_url("sqlite:///x.db") == "sqlite:///x.db"


def test_chat_flow_with_evaluation(client):
    register(client)
    r = client.post("/ask", data={"prompt": "What is an algorithm?"}, follow_redirects=True)
    page = r.data.decode()
    assert "Quick check" in page and page.count('class="tap"') == 4
    iid = int(re.search(r"/respond/(\d+)", page).group(1))
    with client.app.app_context():
        key = db.session.get(Interaction, iid).key      # options are shuffled, so read where the key landed
    r = client.post(f"/respond/{iid}", data={"choice": key, "elapsed_ms": 9000}, follow_redirects=True)
    page = r.data.decode()
    assert "Understood" in page and "evalcard" in page
    with client.app.app_context():
        it = db.session.get(Interaction, iid)
        assert it.correct == 1 and it.w == 1.0 and it.label == "Understood" and it.engine_ms is not None
        assert it.features["level"] == 1 and it.question_words > 0


def test_fast_tap_counts_as_possible_guess(client):
    register(client)
    page = client.post("/ask", data={"prompt": "Explain loops"}, follow_redirects=True).data.decode()
    iid = int(re.search(r"/respond/(\d+)", page).group(1))
    page = client.post(f"/respond/{iid}", data={"choice": 0, "elapsed_ms": 800}, follow_redirects=True).data.decode()
    assert "possible guess" in page
    with client.app.app_context():
        assert db.session.get(Interaction, iid).w == 0.2


def test_must_answer_before_next_prompt(client):
    register(client)
    client.post("/ask", data={"prompt": "first"})
    r = client.post("/ask", data={"prompt": "second"}, follow_redirects=True)
    assert b"answer or skip" in r.data
    with client.app.app_context():
        assert Interaction.query.count() == 1


def test_skip(client):
    register(client)
    page = client.post("/ask", data={"prompt": "x?"}, follow_redirects=True).data.decode()
    iid = int(re.search(r"/respond/(\d+)", page).group(1))
    page = client.post(f"/respond/{iid}", data={"skip": 1}, follow_redirects=True).data.decode()
    assert "skipped" in page


def test_other_users_cannot_answer(client):
    register(client, "alice")
    page = client.post("/ask", data={"prompt": "x?"}, follow_redirects=True).data.decode()
    iid = int(re.search(r"/respond/(\d+)", page).group(1))
    client.get("/logout")
    register(client, "bobby")
    assert client.post(f"/respond/{iid}", data={"choice": 0}).status_code == 404


def test_requires_login(client):
    assert client.get("/chat").status_code == 302


def test_admin_pages_and_exports(client):
    register(client)
    page = client.post("/ask", data={"prompt": "What is a loop?"}, follow_redirects=True).data.decode()
    iid = int(re.search(r"/respond/(\d+)", page).group(1))
    client.post(f"/respond/{iid}", data={"choice": 1, "elapsed_ms": 7000})
    assert client.get("/admin/").status_code == 302
    client.post("/admin/login", data={"password": "pw"})
    assert b"Evaluation log" in client.get("/admin/").data
    csv = client.get("/admin/export/interactions.csv").data.decode().splitlines()
    assert csv[0].lstrip("﻿").startswith("id,user") and len(csv) == 2
    assert b"Prompt set" in client.get("/admin/prompts").data
    assert b"Simulation study" in client.get("/admin/simulation").data


def test_prompt_set_run_and_reports(client):
    from app import jobs, reports
    items = jobs.read_prompt_csv("No.,Prompt,Domain,Target difficulty\n1,What is X?,CS,Easy\n2,What is Y?,Math,Hard\n")
    assert [i["number"] for i in items] == [1, 2]
    jobs.run_prompt_set(client.app, items, "t")
    jobs.run_prompt_set(client.app, items, "t")              # second run skips finished prompts
    with client.app.app_context():
        runs = reports.prompt_runs("t")
        assert len(runs) == 2 and all(r.question_ok for r in runs)
        t3 = reports.table3(runs)
        assert t3["accepted_pct"] == 100 and t3["valid_output_pct"] == 100
        assert reports.review_frame(runs).shape[0] == 2
        assert PromptRun.query.count() == 2

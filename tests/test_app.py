"""End-to-end test: one user goes through the whole sitting with the Flask test client."""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import create_app  # noqa: E402
from app.models import Check, User, db  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    app = create_app({"SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path}/t.db", "TESTING": True,
                      "TESTING_NO_CSRF": True, "ADMIN_PASSWORD": "pw"})
    with app.test_client() as c:
        c.app = app
        yield c


def walk_sitting(c, name="tester", pick=0):
    r = c.post("/register", data={"username": name, "password": "secret1", "password2": "secret1"})
    assert r.status_code == 302 and r.location.endswith("/consent")
    c.post("/consent", data={"agree": "yes"})
    c.post("/profile", data={"year_level": "3", "prog_courses": "3", "ai_courses": "1"})
    from app.views import prior_items
    c.post("/prior", data={it["item_id"]: it["answer"] for it in prior_items()})
    c.post("/tutorial", data={})
    for rnd in range(12):
        r = c.post("/chat", data={"text": "Will I pass if I study 12 hours a week and have 2 absences?"})
        assert b"Here is what I understood" in r.data
        r = c.post("/ask", data={"text": "q", "study_hours": 12, "absences": 2, "sleep_hours": 7,
                                 "assignments_done": 85, "extracurricular_hours": 4, "prev_grade": 84})
        assert r.status_code == 302 and "/answer/" in r.location
        for _ in range(2):
            page = c.get(r.location).data.decode()
            cid = re.search(r"/check/(\d+)", page).group(1)
            r2 = c.post(f"/check/{cid}", data={"choice": pick, "confidence": 4, "trust": 3})
            assert r2.status_code == 302
    r = c.get("/final")
    assert "/answer/" in r.location
    for _ in range(6):
        page = c.get(r.location).data.decode()
        cid = re.search(r"/check/(\d+)", page).group(1)
        c.post(f"/check/{cid}", data={"choice": pick, "confidence": 3, "trust": 3})
    return c.get("/results")


def test_full_sitting(client):
    r = walk_sitting(client)
    assert r.status_code == 200 and b"Thank you for taking part" in r.data
    with client.app.app_context():
        u = User.query.one()
        assert u.stage == "done" and u.round == 12 and u.post_score is not None
        main = Check.query.filter_by(user_id=u.id, phase="main").all()
        assert len(main) == 24 and len({c.item_id for c in main}) == 24
        assert {c.skill for c in main} == {"SHAP", "LIME", "CF"}
        assert sorted(c.order for c in main) == list(range(1, 25))
        assert Check.query.filter_by(user_id=u.id, phase="final").count() == 6
        assert Check.query.filter_by(user_id=u.id, phase="prior").count() == 10


def test_admin_export(client):
    walk_sitting(client)
    assert client.get("/admin/").status_code == 302
    client.post("/admin/login", data={"password": "pw"})
    assert b"Researcher dashboard" in client.get("/admin/").data
    csv = client.get("/admin/export/responses.csv").data.decode().splitlines()
    assert csv[0].startswith("user_id,session,order,item_id,skill,complexity,correct,confidence,trust")
    assert len(csv) == 25
    users = client.get("/admin/export/users.csv").data.decode().splitlines()
    assert users[0].startswith("user_id,prior_knowledge,year_level,ai_courses,post_score") and len(users) == 2


def test_parser_reads_numbers():
    from app.predictor import parse_question
    inp, found = parse_question("Will I pass if I study six hours a week and have 3 absences?")
    assert inp["study_hours"] == 6 and inp["absences"] == 3 and set(found) == {"study_hours", "absences"}
    inp, _ = parse_question("I study 2 hours per day")
    assert inp["study_hours"] == 10


def test_cannot_skip_ahead(client):
    client.post("/register", data={"username": "skipper", "password": "secret1", "password2": "secret1"})
    r = client.get("/chat")
    assert r.status_code == 302 and r.location.endswith("/consent")


def test_database_url_uses_installed_postgres_driver():
    from app import database_url
    assert database_url("postgres://u:p@host/db") == "postgresql+psycopg2://u:p@host/db"
    assert database_url("postgresql://u:p@host/db?sslmode=require") == "postgresql+psycopg2://u:p@host/db?sslmode=require"
    assert database_url("postgresql+psycopg2://u@h/d") == "postgresql+psycopg2://u@h/d"
    assert database_url("sqlite:///x.db") == "sqlite:///x.db"

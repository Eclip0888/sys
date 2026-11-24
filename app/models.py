"""Database tables (SQLite locally, PostgreSQL online)."""
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(40), unique=True, nullable=False)
    pw_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=now)

    @property
    def code(self):
        return f"U{self.id:04d}"


class Interaction(db.Model):
    """One prompt, its AI answer, the generated question, the one-tap response and the evaluation."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), index=True)
    created_at = db.Column(db.DateTime, default=now)
    prompt = db.Column(db.Text)
    answer = db.Column(db.Text)
    answer_seconds = db.Column(db.Float)
    features = db.Column(db.JSON)              # fkgl, len, tech, level, words
    # generated question
    question_ok = db.Column(db.Boolean, default=False)
    question = db.Column(db.Text)
    options = db.Column(db.JSON)
    key = db.Column(db.Integer)
    level = db.Column(db.Integer)
    attempts = db.Column(db.Integer)
    question_words = db.Column(db.Integer)
    qgen_error = db.Column(db.Text)
    shown_at = db.Column(db.DateTime)
    # response
    chosen = db.Column(db.Integer)
    correct = db.Column(db.Integer)
    seconds = db.Column(db.Float)
    skipped = db.Column(db.Boolean, default=False)
    answered_at = db.Column(db.DateTime)
    # evaluation
    b = db.Column(db.Float)
    w = db.Column(db.Float)
    theta = db.Column(db.Float)
    score = db.Column(db.Float)
    label = db.Column(db.String(30))
    reason = db.Column(db.Text)
    engine_ms = db.Column(db.Float)


class PromptRun(db.Model):
    """One prompt from the researchers' prompt set, sent through the system (Chapter 3, Data Generation)."""
    id = db.Column(db.Integer, primary_key=True)
    batch = db.Column(db.String(40), index=True)
    number = db.Column(db.Integer)
    domain = db.Column(db.String(80))
    target = db.Column(db.String(20))
    prompt = db.Column(db.Text)
    answer = db.Column(db.Text)
    features = db.Column(db.JSON)
    question_ok = db.Column(db.Boolean)
    question = db.Column(db.Text)
    options = db.Column(db.JSON)
    key = db.Column(db.Integer)
    level = db.Column(db.Integer)
    attempts = db.Column(db.Integer)
    valid_outputs = db.Column(db.Integer)
    first_try = db.Column(db.Boolean)
    question_words = db.Column(db.Integer)
    answer_seconds = db.Column(db.Float)
    question_seconds = db.Column(db.Float)
    error = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=now)


class Param(db.Model):
    """Key-value store: live algorithm parameters, job status, simulation reports."""
    key = db.Column(db.String(40), primary_key=True)
    value = db.Column(db.JSON)
    updated_at = db.Column(db.DateTime, default=now, onupdate=now)


def get_param(key, default=None):
    p = db.session.get(Param, key)
    return p.value if p is not None else default


def set_param(key, value):
    p = db.session.get(Param, key) or Param(key=key)
    p.value = value
    db.session.add(p)
    db.session.commit()

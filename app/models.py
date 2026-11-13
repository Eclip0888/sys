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
    consent_at = db.Column(db.DateTime)
    # profile (Table 2)
    age = db.Column(db.String(20))
    sex = db.Column(db.String(30))
    program = db.Column(db.String(60))
    year_level = db.Column(db.Integer)
    prog_courses = db.Column(db.Integer)
    ai_courses = db.Column(db.Integer)
    ai_use = db.Column(db.String(30))
    # progress
    stage = db.Column(db.String(20), default="consent")   # consent, profile, prior, tutorial, chat, final, done
    round = db.Column(db.Integer, default=0)                 # completed rounds (0-12)
    plan_seed = db.Column(db.Integer)
    prior_score = db.Column(db.Integer)                      # 0-10
    post_score = db.Column(db.Float)                         # 0-1
    finished_at = db.Column(db.DateTime)
    excluded = db.Column(db.Boolean, default=False)

    @property
    def code(self):
        return f"U{self.id:04d}"


class Ask(db.Model):
    """One question to the AI and its answer (one round)."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), index=True)
    round = db.Column(db.Integer)            # 1-12, or 99 for the final check
    text = db.Column(db.Text)
    inputs = db.Column(db.JSON)
    parsed = db.Column(db.JSON)              # keys read from the text
    label = db.Column(db.String(10))
    prob_pass = db.Column(db.Float)
    xai_type = db.Column(db.String(10))      # SHAP, LIME, CF or ALL (final)
    explanation = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=now)


class Check(db.Model):
    """One comprehension question (generated from a template) and the user's answer."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), index=True)
    ask_id = db.Column(db.Integer, db.ForeignKey("ask.id"), index=True)
    phase = db.Column(db.String(10))         # prior, main, final
    order = db.Column(db.Integer)            # position within the phase
    item_id = db.Column(db.String(10))
    skill = db.Column(db.String(10))
    complexity = db.Column(db.String(2))
    prompt = db.Column(db.Text)
    options = db.Column(db.JSON)
    answer = db.Column(db.Integer)
    chosen = db.Column(db.Integer)
    correct = db.Column(db.Integer)
    confidence = db.Column(db.Integer)
    trust = db.Column(db.Integer)
    shown_at = db.Column(db.DateTime, default=now)
    answered_at = db.Column(db.DateTime)


class Param(db.Model):
    """Calibrated model parameters (IRT, BKT, Random Forest) as JSON or bytes."""
    key = db.Column(db.String(20), primary_key=True)
    value = db.Column(db.JSON)
    blob = db.Column(db.LargeBinary)
    updated_at = db.Column(db.DateTime, default=now, onupdate=now)

"""XAI-Check: an AI Assistant that asks users whether they understood its explanations."""
import os
import secrets

from flask import Flask, abort, request, session

from .models import db

_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        from .predictor import Predictor
        _predictor = Predictor(cache_dir=os.environ.get("MODEL_CACHE_DIR"))
    return _predictor


def database_url(url: str) -> str:
    """Make a PostgreSQL URL from Neon, Supabase or Render use the installed psycopg2 driver.

    Providers give "postgres://..." or "postgresql://...".  SQLAlchemy 2.1 maps a plain
    "postgresql://" to the psycopg (v3) driver, which is not installed, so the driver is
    named explicitly.
    """
    url = url.strip()
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg2://" + url[len(prefix):]
    return url


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    os.makedirs(app.instance_path, exist_ok=True)
    url = database_url(os.environ.get("DATABASE_URL", "sqlite:///" + os.path.join(app.instance_path, "xai_check.db")))
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-only-change-me"),
        SQLALCHEMY_DATABASE_URI=url,
        SQLALCHEMY_ENGINE_OPTIONS={"pool_pre_ping": True},
        ADMIN_PASSWORD=os.environ.get("ADMIN_PASSWORD", "admin"),
        STUDY_NAME=os.environ.get("STUDY_NAME", "XAI-Check"),
    )
    if test_config:
        app.config.update(test_config)
    db.init_app(app)

    @app.before_request
    def csrf_protect():
        if request.method == "POST" and not app.config.get("TESTING_NO_CSRF"):
            if request.form.get("csrf") != session.get("csrf"):
                abort(400, "Your session expired. Please go back and try again.")

    @app.context_processor
    def inject():
        if "csrf" not in session:
            session["csrf"] = secrets.token_urlsafe(16)
        return {"csrf": session["csrf"], "study_name": app.config["STUDY_NAME"]}

    from .views import bp as study_bp
    from .admin import bp as admin_bp
    app.register_blueprint(study_bp)
    app.register_blueprint(admin_bp)

    with app.app_context():
        db.create_all()
    return app

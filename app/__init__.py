"""AI chatbot with real-time evaluation of user comprehension of AI outputs."""
import os
import secrets

from flask import Flask, abort, request, session

from .models import db


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
    url = database_url(os.environ.get("DATABASE_URL", "sqlite:///" + os.path.join(app.instance_path, "chatbot.db")))
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-only-change-me"),
        SQLALCHEMY_DATABASE_URI=url,
        SQLALCHEMY_ENGINE_OPTIONS={"pool_pre_ping": True},
        ADMIN_PASSWORD=os.environ.get("ADMIN_PASSWORD", "admin"),
        APP_NAME=os.environ.get("APP_NAME", "CompreCheck AI"),
        MAX_CONTENT_LENGTH=2 * 1024 * 1024,
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
        from .gemini import get_client
        return {"csrf": session["csrf"], "app_name": app.config["APP_NAME"], "offline_ai": get_client().fake}

    from .views import bp as chat_bp
    from .admin import bp as admin_bp
    app.register_blueprint(chat_bp)
    app.register_blueprint(admin_bp)

    with app.app_context():
        db.create_all()
    return app

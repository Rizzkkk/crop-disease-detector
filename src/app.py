from __future__ import annotations

from pathlib import Path

from flask import Flask
from flask_login import LoginManager

from .db import db
from .auth.models import User
from .auth.routes import auth_bp
from .scans.routes import scans_bp


login_manager = LoginManager()


def create_app() -> Flask:
    app = Flask(__name__)

    project_root = Path(__file__).resolve().parents[1]
    app.config.update(
        SECRET_KEY="dev-secret-key-change-me",
        SQLALCHEMY_DATABASE_URI="sqlite:///app.db",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        UPLOAD_FOLDER=str(project_root / "uploads"),
    )

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"

    @login_manager.user_loader
    def load_user(user_id: str):
        if not user_id:
            return None
        return User.query.get(int(user_id))

    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(scans_bp, url_prefix="/scans")

    return app


app = create_app()


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)


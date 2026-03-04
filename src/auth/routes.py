from __future__ import annotations

from flask import Blueprint, jsonify, request
from flask_login import LoginManager, current_user, login_required, login_user, logout_user

from ..db import db
from .models import User


auth_bp = Blueprint("auth", __name__)


@auth_bp.post("/register")
def register():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "email and password required"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "email already registered"}), 400

    user = User(email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    return jsonify({"id": user.id, "email": user.email}), 201


@auth_bp.post("/login")
def login():
    if current_user.is_authenticated:
        return jsonify({"message": "already logged in", "email": current_user.email})

    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify({"error": "invalid credentials"}), 401

    login_user(user)
    return jsonify({"message": "logged in", "email": user.email})


@auth_bp.post("/logout")
@login_required
def logout():
    logout_user()
    return jsonify({"message": "logged out"})


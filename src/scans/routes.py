from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from flask import Blueprint, jsonify, current_app, request
from flask_login import current_user, login_required

from .service import create_scan, delete_scan, get_scan, list_scans, update_scan


scans_bp = Blueprint("scans", __name__)


@scans_bp.post("/scan")
@login_required
def scan():
    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify({"error": "image file required"}), 400

    upload_folder = Path(current_app.config.get("UPLOAD_FOLDER", "uploads"))
    upload_folder.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix or ".jpg"
    dest = upload_folder / f"upload_{current_user.id}_{file.mimetype.replace('/', '_')}{ext}"
    file.save(dest)

    record = create_scan(str(dest), user_id=current_user.id)

    return (
        jsonify(
            {
                "id": record.id,
                "image_path": record.image_path,
                "prediction": record.prediction,
                "confidence": record.confidence,
                "created_at": record.created_at.isoformat(),
                "user_id": record.user_id,
            }
        ),
        201,
    )


@scans_bp.get("/history")
@login_required
def history():
    records = list_scans(user_id=current_user.id)
    return jsonify(
        [
            {
                "id": r.id,
                "image_path": r.image_path,
                "prediction": r.prediction,
                "confidence": r.confidence,
                "created_at": r.created_at.isoformat(),
                "user_id": r.user_id,
            }
            for r in records
        ]
    )


@scans_bp.put("/history/<int:scan_id>")
@login_required
def history_update(scan_id: int):
    fields: Dict[str, Any] = request.get_json(silent=True) or {}
    record = update_scan(scan_id, fields, user_id=current_user.id)
    if not record:
        return jsonify({"error": "not found"}), 404

    return jsonify(
        {
            "id": record.id,
            "image_path": record.image_path,
            "prediction": record.prediction,
            "confidence": record.confidence,
            "created_at": record.created_at.isoformat(),
            "user_id": record.user_id,
        }
    )


@scans_bp.delete("/history/<int:scan_id>")
@login_required
def history_delete(scan_id: int):
    ok = delete_scan(scan_id, user_id=current_user.id)
    if not ok:
        return jsonify({"error": "not found"}), 404
    return jsonify({"deleted": True})


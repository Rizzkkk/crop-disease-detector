from __future__ import annotations

from datetime import datetime, timezone

from ..db import db


class ScanRecord(db.Model):
    __tablename__ = "scan_records"

    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(512), nullable=False)
    prediction = db.Column(db.String(128), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(
        db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    user_id = db.Column(db.Integer, nullable=True)


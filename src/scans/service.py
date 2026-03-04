from __future__ import annotations

import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import current_app

from ..db import db
from ..ml.inference import ModelPaths, load_labels, load_tf_model, predict
from ..ml.preprocess import PreprocessConfig, preprocess_image
from .models import ScanRecord


def get_uploads_dir() -> Path:
    # Get the uploads directory, using Flask config when available.
    upload_base = None
    if current_app and current_app.config.get("UPLOAD_FOLDER"):
        upload_base = Path(current_app.config["UPLOAD_FOLDER"])
    else:
        # Fallback to a folder named 'uploads' next to this file's root project
        project_root = Path(__file__).resolve().parents[2]
        upload_base = project_root / "uploads"

    upload_base.mkdir(parents=True, exist_ok=True)
    return upload_base


def save_uploaded_image(source_path: str) -> str:
    # Copy the image into uploads/ with a unique filename and return its path.
    uploads_dir = get_uploads_dir()

    src = Path(source_path)
    if not src.is_file():
        raise FileNotFoundError(f"Image file not found: {source_path}")

    ext = src.suffix or ".jpg"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    dest = uploads_dir / unique_name

    shutil.copy2(src, dest)

    return str(dest)


def run_scan(image_path: str) -> Dict[str, Any]:
    # Run the ML prediction pipeline on an image and return a result dict.
    model_paths = ModelPaths()
    labels = load_labels(model_paths.labels_path)
    model = load_tf_model(model_paths.model_path)

    cfg = PreprocessConfig()
    x = preprocess_image(image_path, cfg)

    label, confidence, class_idx = predict(model, x, labels)

    # Load recommendations JSON
    reco_path = Path("data") / "recommendations.json"
    full_reco_path = Path(__file__).resolve().parents[2] / reco_path
    recommendation: Optional[dict] = None
    if full_reco_path.is_file():
        import json

        with full_reco_path.open("r", encoding="utf-8") as f:
            recommendations = json.load(f)
        recommendation = recommendations.get(label)

    return {
        "label": label,
        "confidence": float(confidence),
        "class_index": int(class_idx),
        "recommendation": recommendation,
    }


def save_scan(
    image_path: str,
    prediction: str,
    confidence: float,
    user_id: Optional[int] = None,
) -> ScanRecord:
    # Persist a scan record to the database.
    stored_path = save_uploaded_image(image_path)

    record = ScanRecord(
        image_path=stored_path,
        prediction=prediction,
        confidence=confidence,
        user_id=user_id,
        created_at=datetime.now(timezone.utc),
    )
    db.session.add(record)
    db.session.commit()
    return record


def create_scan(
    image_path: str,
    user_id: Optional[int] = None,
) -> ScanRecord:
    # Run the full scan pipeline and save the result as a ScanRecord.
    result = run_scan(image_path)
    return save_scan(
        image_path=image_path,
        prediction=result["label"],
        confidence=result["confidence"],
        user_id=user_id,
    )


def list_scans(user_id: Optional[int] = None) -> List[ScanRecord]:
    query = ScanRecord.query
    if user_id is not None:
        query = query.filter_by(user_id=user_id)
    return query.order_by(ScanRecord.created_at.desc()).all()


def get_scan(scan_id: int, user_id: Optional[int] = None) -> Optional[ScanRecord]:
    query = ScanRecord.query.filter_by(id=scan_id)
    if user_id is not None:
        query = query.filter_by(user_id=user_id)
    return query.first()


def update_scan(
    scan_id: int,
    fields: Dict[str, Any],
    user_id: Optional[int] = None,
) -> Optional[ScanRecord]:
    record = get_scan(scan_id, user_id=user_id)
    if not record:
        return None

    for key, value in fields.items():
        if hasattr(record, key):
            setattr(record, key, value)
    db.session.commit()
    return record


def delete_scan(scan_id: int, user_id: Optional[int] = None) -> bool:
    record = get_scan(scan_id, user_id=user_id)
    if not record:
        return False

    db.session.delete(record)
    db.session.commit()
    return True


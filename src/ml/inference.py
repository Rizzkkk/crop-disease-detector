from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from tensorflow.keras.models import load_model


@dataclass(frozen=True)
class ModelPaths:
    model_path: str = os.path.join("models", "model.keras")
    labels_path: str = os.path.join("models", "labels.json")


class ModelNotFoundError(RuntimeError):
    pass


def load_labels(labels_path: str) -> Dict[int, str]:
    with open(labels_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # labels.json can be {"0": "Healthy"} or {0: "Healthy"}
    labels: Dict[int, str] = {int(k): str(v) for k, v in raw.items()}
    if not labels:
        raise ValueError("labels.json is empty")
    return labels


def load_tf_model(model_path: str):
    if not os.path.exists(model_path):
        raise ModelNotFoundError(
            f"Missing model file at '{model_path}'.\n"
                  )
    return load_model(model_path)


def predict(model, x: np.ndarray, labels: Dict[int, str]) -> Tuple[str, float, int]:
    #Predict class label and confidence.
    probs = model.predict(x, verbose=0)
    probs = np.asarray(probs).reshape(-1)  
    class_idx = int(np.argmax(probs))
    confidence = float(probs[class_idx])

    label = labels.get(class_idx, f"class_{class_idx}")
    return label, confidence, class_idx

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    # Default to common transfer learning input size
    image_size: Tuple[int, int] = (224, 224)
    normalize: bool = True


def preprocess_image(image_path: str, cfg: PreprocessConfig | None = None) -> np.ndarray:
    cfg = cfg or PreprocessConfig()

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, cfg.image_size, interpolation=cv2.INTER_AREA)

    x = img_resized.astype(np.float32)
    if cfg.normalize:
        x = x / 255.0

    # Add batch dimension
    x = np.expand_dims(x, axis=0)
    return x

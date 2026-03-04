from __future__ import annotations

import argparse
import json
import os
import sys

from src.ml.preprocess import preprocess_image, PreprocessConfig
from src.ml.inference import ModelPaths, load_labels, load_tf_model, predict, ModelNotFoundError


def load_recommendations(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Crop Disease Detector CLI")
    parser.add_argument("image", help="Path to a leaf image (jpg/png)")
    parser.add_argument("--model", default=ModelPaths().model_path, help="Path to model.keras")
    parser.add_argument("--labels", default=ModelPaths().labels_path, help="Path to labels.json")
    parser.add_argument("--reco", default=os.path.join("data", "recommendations.json"), help="Path to recommendations.json")
    parser.add_argument("--size", default="224,224", help="Image size H,W (default: 224,224)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}", file=sys.stderr)
        return 2

    try:
        h, w = [int(x.strip()) for x in args.size.split(",")]
    except Exception:
        print("ERROR: --size must be like 224,224", file=sys.stderr)
        return 2

    try:
        labels = load_labels(args.labels)
        model = load_tf_model(args.model)
        x = preprocess_image(args.image, PreprocessConfig(image_size=(w, h)))
        label, confidence, _ = predict(model, x, labels)

        recos = load_recommendations(args.reco)
        recommendation = recos.get(label, "No recommendation found for this class.")

        print(f"Disease detected: {label}")
        print(f"Confidence: {confidence * 100:.2f}%\n")
        print("Recommendation:")
        print(recommendation)
        return 0

    except ModelNotFoundError as e:
        print("ERROR: Model file missing.\n" + str(e), file=sys.stderr)
        return 3
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

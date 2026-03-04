from __future__ import annotations

from pathlib import Path

from src.ml.inference import ModelPaths, load_labels, load_tf_model, predict
from src.ml.preprocess import PreprocessConfig, preprocess_image


def main() -> int:
    sample_path = Path("sample_images") / "rice_blast.jpg"
    if not sample_path.is_file():
        print(f"[SMOKE] Sample image not found at {sample_path!s}")
        return 1

    paths = ModelPaths()
    try:
        labels = load_labels(paths.labels_path)
        model = load_tf_model(paths.model_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[SMOKE] Failed to load model or labels: {exc}")
        return 1

    try:
        x = preprocess_image(str(sample_path), PreprocessConfig())
        label, confidence, _ = predict(model, x, labels)
    except Exception as exc:  # noqa: BLE001
        print(f"[SMOKE] Failed during prediction: {exc}")
        return 1

    print(f"[SMOKE] OK - {label} ({confidence * 100:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Crop Disease Detector

End-to-end project that detects crop leaf diseases using a TensorFlow model, exposes a CLI and Flask API, and stores scan history in a database.

## Setup

- **Python**: 3.11 recommended
- **Install dependencies**:

```bash
pip install -r requirements.txt
```

## CLI Usage

Run a prediction on a local image (from project root):

```bash
python -m scripts.predict sample_images/rice_blast.jpg
```

This will load `models/model.keras` and `models/labels.json`, run inference, and print:

- predicted label
- confidence
- textual recommendation from `data/recommendations.json`

## Flask Backend

Start the backend server:

```bash
python -m src.app
```

The API will listen on `http://127.0.0.1:5000`.

### Auth endpoints

- `POST /auth/register` – JSON `{ "email": "...", "password": "..." }`
- `POST /auth/login` – JSON `{ "email": "...", "password": "..." }`
- `POST /auth/logout`

### Scan endpoints (require login)

- `POST /scans/scan` – multipart form-data with `image=@path/to/image.jpg`
- `GET /scans/history` – list user scan records
- `PUT /scans/history/<id>` – update selected fields
- `DELETE /scans/history/<id>` – delete a scan

## Training Notebook

The model is trained in Colab using:

- `notebooks/01_train_model_colab.ipynb`

This notebook:

- downloads the Kaggle plant disease dataset
- trains a MobileNetV2-based classifier
- exports:
  - `models/model.keras`
  - `models/labels.json`

> Running the app does **not** require the notebook; it is only for training and reproducibility.

## Smoke Test

After setup, you can run a quick smoke test:

```bash
python -m scripts.predict sample_images/rice_blast.jpg
```

If this prints a prediction and confidence instead of an error, the model and dependencies are correctly installed.


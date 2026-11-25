# Batik Checker — FastAPI backend

REST API that classifies Indonesian batik motifs from an uploaded image.
It reuses the exact preprocessing + feature pipeline from the training notebook.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check / API info |
| GET | `/motifs` | List supported motif classes + descriptions |
| POST | `/predict` | Upload an image, get the top-3 predictions |
| GET | `/docs` | Auto-generated interactive API docs (Swagger UI) |

## Setup

From the project root, activate the venv, then install backend deps:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

## Run

```powershell
cd backend
..\.venv\Scripts\uvicorn.exe main:app --reload --port 8000
```

Then open http://localhost:8000/docs to test the API from the browser.

## Example request

```powershell
curl -X POST http://localhost:8000/predict -F "file=@..\data\selected\parang\1.jpg"
```

Example response:

```json
{
  "predictions": [
    {"label": "parang", "probability": 0.84},
    {"label": "ceplok", "probability": 0.08},
    {"label": "kawung", "probability": 0.04}
  ],
  "all_probabilities": {"parang": 0.84, "...": "..."},
  "feature_length": 114,
  "images": {"input": "data:image/png;base64,...", "grayscale": "data:image/png;base64,..."}
}
```

## Files

- `main.py` — FastAPI app, endpoints, request validation, CORS.
- `classifier.py` — preprocessing + feature extraction + prediction (`BatikClassifier`).

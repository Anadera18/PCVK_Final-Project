"""Batik Checker - FastAPI backend.

Endpoints:
    GET  /            -> health check / API info
    GET  /motifs      -> list of supported motif classes + descriptions
    POST /predict     -> upload an image, get the top-k predicted motifs

Run from the backend/ folder:

    uvicorn main:app --reload --port 8000

Interactive API docs are then available at http://localhost:8000/docs
"""

import base64
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from classifier import BatikClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
TOP_K = 3

MOTIF_INFO = {
    "parang": "Diagonal geometric motif with repeated S-like lines.",
    "kawung": "Radial geometric motif with dense symmetric ovals.",
    "megamendung": "Cloud/curve motif, traditionally dominated by blue.",
    "cendrawasih": "Organic floral-fauna (bird of paradise) motif, very colourful.",
    "ceplok": "Centred geometric rosette with dense symmetric filling.",
}

classifier = BatikClassifier()


# ---------------------------------------------------------------------------
# Lifespan: load the model once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        classifier.load()
        print("Model loaded successfully.")
    except FileNotFoundError as exc:
        print(f"WARNING: could not load model ({exc}). "
              f"Train the notebook first to create files in models/.")
    yield


app = FastAPI(
    title="Batik Checker API",
    description="Classifies Indonesian batik motifs using classical machine learning.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS so the React frontend (different port) can call this API during dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictionItem(BaseModel):
    label: str
    probability: float


class PredictResponse(BaseModel):
    predictions: list[PredictionItem]
    all_probabilities: dict[str, float]
    feature_length: int
    images: dict[str, str]  # name -> base64 PNG data URL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_data_url(bgr_image):
    """Encode a single-channel or BGR image as a base64 PNG data URL."""
    ok, buffer = cv2.imencode(".png", bgr_image)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image.")
    b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "name": "Batik Checker API",
        "status": "ready" if classifier.is_ready() else "model not loaded",
        "endpoints": ["/", "/motifs", "/predict", "/docs"],
    }


@app.get("/motifs")
def motifs():
    return {"motifs": [{"name": name, "description": desc}
                       for name, desc in MOTIF_INFO.items()]}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not classifier.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the training notebook to create models/.",
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 5 MB).")

    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not read the image.")

    result = classifier.predict(img, top_k=TOP_K)
    processed = result["processed"]

    images = {
        "input": to_data_url(processed["resized"]),
        "grayscale": to_data_url(processed["gray"]),
    }

    return PredictResponse(
        predictions=result["predictions"],
        all_probabilities=result["all_probabilities"],
        feature_length=result["feature_length"],
        images=images,
    )

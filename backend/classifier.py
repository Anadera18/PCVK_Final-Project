"""Feature extraction and prediction pipeline.

This module is the bridge between the trained model (in ../models) and the
FastAPI backend. It reuses the EXACT same preprocessing and feature pipeline as
the training notebook (`notebooks/batik_checker.ipynb`):

    resize 128x128 -> grayscale + HSV -> Gaussian blur -> histogram equalization
    features = GLCM + LBP + HSV histogram (114 dims) -> StandardScaler
"""

from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# ---------------------------------------------------------------------------
# Paths & configuration (must match the notebook)
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

IMG_SIZE = (128, 128)
GAUSSIAN_KERNEL = (5, 5)
GAUSSIAN_SIGMA = 0

GLCM_DISTANCES = [1, 2]
GLCM_ANGLES = [0, 45, 90, 135]  # degrees
GLCM_LEVELS = 32

LBP_RADIUS = 1
LBP_N_POINTS = 8
LBP_METHOD = "uniform"

HSV_BINS = [4, 4, 4]
HSV_RANGES = [0, 180, 0, 256, 0, 256]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_image(img):
    """Resize -> grayscale + HSV -> Gaussian blur -> histogram equalization.

    Returns a dict with the processed 'gray' image plus the 'hsv' image and the
    'resized' BGR image (useful for returning intermediate images to the UI).
    """
    resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
    gray = cv2.equalizeHist(gray)
    return {"gray": gray, "hsv": hsv, "resized": resized}


# ---------------------------------------------------------------------------
# Feature extraction (GLCM + LBP + HSV)
# ---------------------------------------------------------------------------
def glcm_features(gray):
    """Haralick texture features from the Gray-Level Co-occurrence Matrix."""
    quant = (gray.astype(np.float32) / 256.0 * GLCM_LEVELS).astype(np.uint8)
    quant = np.clip(quant, 0, GLCM_LEVELS - 1)
    angles = [np.deg2rad(a) for a in GLCM_ANGLES]
    glcm = graycomatrix(quant, distances=GLCM_DISTANCES, angles=angles,
                        levels=GLCM_LEVELS, symmetric=True, normed=True)
    feats = []
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        feats.extend(graycoprops(glcm, prop).flatten())
    return np.asarray(feats, dtype=np.float32)


def lbp_features(gray):
    """Normalised Local Binary Pattern histogram."""
    lbp = local_binary_pattern(gray, P=LBP_N_POINTS, R=LBP_RADIUS, method=LBP_METHOD)
    n_bins = LBP_N_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist


def hsv_histogram_features(hsv):
    """Normalised HSV colour histogram (4x4x4)."""
    hist = cv2.calcHist([hsv], [0, 1, 2], None, HSV_BINS, HSV_RANGES)
    hist = hist.flatten().astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist


def extract_features(gray, hsv):
    """Concatenate GLCM + LBP + HSV histogram into a single feature vector."""
    return np.concatenate([
        glcm_features(gray),
        lbp_features(gray),
        hsv_histogram_features(hsv),
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# Model loading & prediction
# ---------------------------------------------------------------------------
class BatikClassifier:
    """Loads the trained model once and predicts on batik images."""

    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.model = None
        self.scaler = None
        self.labels = None

    def load(self):
        """Load model, scaler, and label list from disk."""
        self.model = joblib.load(self.models_dir / "best_model.joblib")
        self.scaler = joblib.load(self.models_dir / "scaler.joblib")
        self.labels = joblib.load(self.models_dir / "labels.joblib")
        return self

    def is_ready(self):
        return self.model is not None and self.scaler is not None and self.labels is not None

    def predict(self, img, top_k=3):
        """Predict a single BGR image.

        Returns a dict with the sorted top-k predictions, the processed
        intermediate images, and the feature length.
        """
        if not self.is_ready():
            raise RuntimeError("Model is not loaded. Call load() first.")

        processed = preprocess_image(img)
        vec = extract_features(processed["gray"], processed["hsv"])
        vec_scaled = self.scaler.transform(vec.reshape(1, -1))
        probs = self.model.predict_proba(vec_scaled)[0]

        order = np.argsort(probs)[::-1]
        predictions = [
            {"label": self.labels[i], "probability": float(probs[i])}
            for i in order
        ]
        return {
            "predictions": predictions[:top_k],
            "all_probabilities": {self.labels[i]: float(probs[i]) for i in range(len(self.labels))},
            "feature_length": int(vec.shape[0]),
            "processed": processed,
        }

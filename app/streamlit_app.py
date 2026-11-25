"""Batik Checker - Streamlit web app (self-contained).

Semua logika preprocessing dan ekstraksi fitur ada di file ini, SAMA PERSIS
dengan yang ada di notebook `notebooks/batik_checker.ipynb`. Model, scaler, dan
label dibaca dari folder `models/` yang dihasilkan notebook.

Jalankan dari root project:

    .\.venv\Scripts\streamlit.exe run app/streamlit_app.py
"""

from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# ---------------------------------------------------------------------------
# Konfigurasi (sama dengan notebook)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

IMG_SIZE = (128, 128)
GAUSSIAN_KERNEL = (5, 5)
GAUSSIAN_SIGMA = 0
GLCM_DISTANCES = [1, 2]
GLCM_ANGLES = [0, 45, 90, 135]
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
    """Resize -> grayscale + HSV -> Gaussian blur -> histogram equalization."""
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
    gray = cv2.equalizeHist(gray)
    return {"gray": gray, "hsv": hsv}


# ---------------------------------------------------------------------------
# Feature extraction (GLCM + LBP + HSV)
# ---------------------------------------------------------------------------
def glcm_features(gray):
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
    lbp = local_binary_pattern(gray, P=LBP_N_POINTS, R=LBP_RADIUS, method=LBP_METHOD)
    n_bins = LBP_N_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist


def hsv_histogram_features(hsv):
    hist = cv2.calcHist([hsv], [0, 1, 2], None, HSV_BINS, HSV_RANGES)
    hist = hist.flatten().astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist


def extract_features(gray, hsv):
    return np.concatenate([
        glcm_features(gray),
        lbp_features(gray),
        hsv_histogram_features(hsv),
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Batik Checker", page_icon="🧵", layout="centered")

st.title("🧵 Batik Checker")
st.write(
    "Unggah foto motif batik, lalu sistem akan menebak jenis motifnya "
    "menggunakan fitur GLCM, LBP, dan histogram warna HSV dengan model "
    "machine learning klasik (SVM / Random Forest)."
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    labels = joblib.load(MODELS_DIR / "labels.joblib")
    return model, scaler, labels


try:
    clf, scaler, label_names = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model belum ada. Jalankan notebook `notebooks/batik_checker.ipynb` "
        "sampai cell terakhir untuk melatih dan menyimpan model."
    )
    st.stop()

uploaded = st.file_uploader(
    "Pilih gambar batik", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Gambar tidak bisa dibaca. Coba file lain.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                 caption="Gambar masukan", use_container_width=True)

    processed = preprocess_image(img)
    with col2:
        st.image(processed["gray"],
                 caption="Hasil preprocessing (grayscale + equalization)",
                 use_container_width=True)

    vec = extract_features(processed["gray"], processed["hsv"])
    vec = scaler.transform(vec.reshape(1, -1))
    probs = clf.predict_proba(vec)[0]
    idx = int(np.argmax(probs))
    label = label_names[idx]
    confidence = float(probs[idx])

    st.success(f"**Prediksi: {label.upper()}** — confidence {confidence:.2%}")

    st.subheader("Probabilitas tiap kelas")
    df = pd.DataFrame({"Kelas": label_names, "Probabilitas": probs})
    df = df.sort_values("Probabilitas", ascending=False)
    st.bar_chart(df.set_index("Kelas"))
    st.dataframe(
        df.assign(Probabilitas=lambda d: d["Probabilitas"].map("{:.2%}".format)),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Mata kuliah Pengolahan Citra dan Visi Komputer — "
    "model klasik (SVM / Random Forest) disimpan dengan joblib."
)

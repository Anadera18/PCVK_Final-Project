# Batik Checker

Website yang menebak **jenis motif batik** dari foto menggunakan machine learning
klasik (tanpa deep learning). Tugas akhir mata kuliah *Pengolahan Citra dan Visi
Komputer*.

## Cara kerja

```
foto → preprocessing → fitur (GLCM + LBP + HSV) → SVM / Random Forest → prediksi motif
```

- **Preprocessing:** resize 128×128 → grayscale + HSV → Gaussian blur → histogram equalization.
- **Fitur:** tekstur GLCM + tekstur lokal LBP + histogram warna HSV (4×4×4), digabung (114 fitur) lalu di-scaling.
- **Model:** SVM dan Random Forest, dibandingkan lewat accuracy, confusion matrix, dan 5-fold cross-validation.
- **Hasil:** Random Forest terbaik (hold-out accuracy ≈ 0.69, 5-fold ≈ 0.62).
- **Website:** **React** frontend + **FastAPI** backend — upload foto, lihat prediksi kelas dan confidence.

## Kelas (5)

`parang`, `kawung`, `megamendung`, `cendrawasih`, `ceplok`.

## Struktur project

```
Final_Project/
├── data/
│   ├── raw/                  # dataset asli 20 kelas (arsip)
│   └── selected/             # 5 kelas terpilih untuk training
├── notebooks/
│   └── batik_checker.ipynb   # SELURUH pipeline ML ada di sini (self-contained)
├── backend/                  # REST API (FastAPI) untuk prediksi
│   ├── main.py               # endpoint: / , /motifs , /predict , /docs
│   ├── classifier.py         # preprocessing + ekstraksi fitur + prediksi
│   └── requirements.txt
├── frontend/                 # web app (React + Vite)
│   ├── src/App.jsx           # halaman upload + hasil
│   ├── src/api.js            # panggil backend
│   └── package.json
├── app/
│   └── streamlit_app.py      # versi lama (Streamlit) — opsional
├── models/                   # model, scaler, label (joblib) hasil notebook
├── results/                  # confusion matrix + metrics.json
├── docs/
│   ├── laporan.md            # draft laporan
│   ├── handoff_gdocs.md      # prompt + laporan untuk Google Docs
│   └── prompt_overview.md    # prompt untuk membangun notebook dengan model lain
├── requirements.txt          # dependency ML (notebook)
└── README.md
```

Semua kode ML ada **di dalam notebook** — tanpa script terpisah. Untuk retraining,
cukup ubah parameter di notebook dan jalankan ulang.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Training / retraining

1. Buka `notebooks/batik_checker.ipynb` (pilih kernel `.venv`).
2. Ubah parameter di bagian **Configuration** atau hiperparameter model di bagian **Training** kalau perlu.
3. **Run All** (menu *Cell → Run All*).

Output yang dihasilkan:

- `models/best_model.joblib`, `models/scaler.joblib`, `models/labels.joblib`
- `results/confusion_matrix_*.png`, `results/cross_validation.png`,
  `results/dataset_samples.png`, `results/preprocessing_steps.png`,
  `results/sample_predictions.png`, `results/metrics.json`

## Menjalankan website (React + FastAPI)

**1. Backend (FastAPI):**

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
cd backend
..\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

Buka http://localhost:8000/docs untuk mencoba API.

**2. Frontend (React):**

```powershell
cd frontend
npm install
npm run dev
```

Buka http://localhost:5173 (butuh backend tetap jalan). Frontend mem-proxy `/api`
ke backend di port 8000.

## Versi lama (Streamlit)

Masih tersedia sebagai alternatif:

```powershell
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py
```

## Catatan

- **Tesseract (OCR) tidak dipakai** — motif batik tidak berisi teks, jadi OCR tidak relevan.
- Model terbaik dipilih otomatis berdasarkan accuracy tertinggi.

# Overview Prompt — Batik Checker Notebook

Use this prompt to have another model (e.g. ChatGPT, Claude) build or rebuild the
training notebook. It describes the exact pipeline used in this project.

---

## Prompt

> Build a Jupyter notebook (`notebooks/batik_checker.ipynb`) for a classical
> computer-vision project called **Batik Checker**. The notebook must classify
> Indonesian batik motifs from photos using **classical machine learning only
> (no deep learning)**.
>
> **Data**
> - Images are organised as `data/selected/<class>/` with 5 classes:
>   `parang`, `kawung`, `megamendung`, `cendrawasih`, `ceplok`.
> - Split the data 80% train / 20% test, **stratified** by class.
>
> **Preprocessing (each image)**
> 1. Resize to 128×128.
> 2. Convert to grayscale and to HSV.
> 3. Gaussian blur (5×5 kernel) to reduce noise.
> 4. Histogram equalization on the grayscale image.
>
> **Feature extraction (concatenate into one vector, then scale)**
> - **GLCM** (texture): distances {1, 2}, angles {0°, 45°, 90°, 135°},
>   32 grey levels; features: contrast, dissimilarity, homogeneity, energy,
>   correlation.
> - **LBP** (local texture): uniform pattern histogram, P=8, R=1, normalised.
> - **HSV colour histogram**: joint 8×8×8 histogram over H, S, V.
> - Scale the concatenated vector with a `StandardScaler` fit on the training
>   set only.
>
> **Models**
> - **SVM** (RBF kernel, C=10) and **Random Forest** (200 trees). Train both and
>   compare.
>
> **Evaluation**
> - Accuracy, per-class precision/recall/F1, and confusion matrix for each model.
>
> **Saving**
> - Save the best model, the scaler, and the label list with `joblib` into
>   `models/`, so a Streamlit app can reuse them.
>
> **Reuse existing code**
> - The notebook must be **fully self-contained**: define all preprocessing,
>   feature extraction, and training code inside notebook cells (do not import
>   from a separate package). Group the tunable settings (image size, class list,
>   GLCM/LBP/HSV parameters, test size, random seed) into one config cell so the
>   user can easily change them and retrain.
> - Add `sys.path`-free, portable path handling: resolve the project root from
>   `Path.cwd()` so it works whether the notebook runs from `notebooks/` or the
>   project root.
>
> Write clear markdown cells between code cells and add comments. Do not use any
> deep-learning library.

---

## Notes

- Tesseract/OCR is intentionally excluded — batik motifs contain no text.
- After training, the notebook saves the best model, scaler and labels with
  `joblib` into `models/`, plus the confusion matrices and `metrics.json` into
  `results/`. The Streamlit app in `app/streamlit_app.py` reuses them.

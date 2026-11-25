import { useState } from "react";
import { predictImage } from "./api";
import { ACCEPTED_TYPES, MAX_FILE_SIZE } from "./config";
import PredictionList from "./components/PredictionList";
import ImagePreview from "./components/ImagePreview";

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  function handleFileChange(event) {
    const selected = event.target.files?.[0];
    setResult(null);
    setError(null);
    if (!selected) return;

    if (!ACCEPTED_TYPES.includes(selected.type)) {
      setError("Unsupported file type. Please choose a JPG, PNG, BMP, or WEBP image.");
      return;
    }
    if (selected.size > MAX_FILE_SIZE) {
      setError("File is too large (max 5 MB).");
      return;
    }

    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
  }

  async function handleSubmit() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predictImage(file);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <h1>Batik Checker</h1>
      <p className="subtitle">
        Upload a batik photo and the system predicts its motif type using
        classical machine learning (GLCM + LBP + HSV features).
      </p>

      <div className="card">
        <input
          type="file"
          accept={ACCEPTED_TYPES.join(",")}
          capture="environment"
          onChange={handleFileChange}
        />
        <p style={{ color: "#52606d", fontSize: "0.85rem" }}>
          Tip: on a phone you can take a photo directly with the camera.
        </p>

        {previewUrl && (
          <img className="preview" src={previewUrl} alt="Selected batik" />
        )}

        <div style={{ marginTop: "1rem" }}>
          <button onClick={handleSubmit} disabled={!file || loading}>
            {loading ? "Predicting..." : "Predict motif"}
          </button>
        </div>

        {error && <p className="error">{error}</p>}
      </div>

      {result && (
        <div className="card">
          <PredictionList predictions={result.predictions} />
          <ImagePreview images={result.images} />
          <p style={{ color: "#52606d", fontSize: "0.8rem", marginTop: "1rem" }}>
            Feature vector length: {result.feature_length}
          </p>
        </div>
      )}
    </div>
  );
}

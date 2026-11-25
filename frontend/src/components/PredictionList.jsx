// Displays the top-k predictions with confidence bars.
export default function PredictionList({ predictions }) {
  if (!predictions || predictions.length === 0) return null;

  return (
    <div>
      <h3>Prediction</h3>
      <ul className="result-list">
        {predictions.map((item) => (
          <li key={item.label}>
            <span style={{ flex: "0 0 120px", textTransform: "capitalize" }}>
              {item.label}
            </span>
            <div style={{ flex: 1, margin: "0 1rem" }}>
              <div
                className="bar"
                style={{ width: `${Math.round(item.probability * 100)}%` }}
              />
            </div>
            <span>{(item.probability * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

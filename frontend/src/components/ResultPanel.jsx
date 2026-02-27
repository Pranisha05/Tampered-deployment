import React from "react";

export default function ResultPanel({ result, previewUrl }) {
  if (!result) return null;

  const label = result.predicted_class;
  const score = result.confidence;

  // Build all_scores array from classes and probs
  const allScores = result.classes.map((c, i) => ({ label: c, score: result.probs[i] || 0 }));

  // Construct image sources from base64 data
  const origSrc = previewUrl;
  const elaSrc = result.ela_base64 ? `data:image/png;base64,${result.ela_base64}` : null;
  const noiseSrc = result.noise_base64 ? `data:image/png;base64,${result.noise_base64}` : null;
  const overlaySrc = result.gradcam_overlay_base64 ? `data:image/png;base64,${result.gradcam_overlay_base64}` : null;
  const heatmapSrc = result.heatmap_base64 ? `data:image/png;base64,${result.heatmap_base64}` : null;
  const maskSrc = result.mask_base64 ? `data:image/png;base64,${result.mask_base64}` : null;

  return (
    <section className="card result-card" style={{ marginTop: 24 }}>
      <div className="result__head" style={{ display: "flex", gap: 16 }}>
        <div className="result__label" style={{ flex: 1 }}>
          <span className={`pill ${label === "authentic" ? "pill--good" : "pill--warn"}`}>
            {label.replace(/_/g, " ").replace(/\b\w/g, (s) => s.toUpperCase())}
          </span>
          <div className="result__score" style={{ marginTop: 8 }}>
            Confidence: {typeof score === "number" ? `${(score * 100).toFixed(2)}%` : "N/A"}
          </div>
        </div>

        <div className="result__detail" style={{ flex: 2 }}>
          <div className="detail__title">Class Probabilities</div>
          <div className="bars" style={{ marginTop: 8 }}>
            {allScores.length === 0 ? (
              <div>No class probabilities returned.</div>
            ) : (
              allScores.map((item) => (
                <div className="bar" key={item.label} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                  <div className="bar__label" style={{ width: 120 }}>
                    {item.label.replace(/_/g, " ")}
                  </div>
                  <div className="bar__track" style={{ flex: 1, background: "#eee", height: 12, borderRadius: 6, overflow: "hidden" }}>
                    <div className="bar__fill" style={{ width: `${(item.score || 0) * 100}%`, height: "100%", background: "#4caf50" }} />
                  </div>
                  <div className="bar__value" style={{ width: 60, textAlign: "right" }}>
                    {(item.score || 0 * 100).toFixed ? `${(item.score * 100).toFixed(2)}%` : `${(item.score * 100).toFixed(2)}%`}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <div className="result__grid" style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginTop: 16 }}>
        <div className="panel">
          <div className="panel__title">Input Image</div>
          {origSrc ? <img src={origSrc} alt="Input image" style={{ width: "100%", borderRadius: 6 }} /> : <div>No original image available</div>}
        </div>
        <div className="panel">
          <div className="panel__title">ELA</div>
          {elaSrc ? <img src={elaSrc} alt="ELA image" style={{ width: "100%", borderRadius: 6 }} /> : <div>No ELA image returned</div>}
        </div>
        <div className="panel">
          <div className="panel__title">Noise Map</div>
          {noiseSrc ? <img src={noiseSrc} alt="Noise map" style={{ width: "100%", borderRadius: 6 }} /> : <div>No noise map returned</div>}
        </div>

        <div className="panel">
          <div className="panel__title">Suspected Area (Overlay)</div>
          {overlaySrc ? <img src={overlaySrc} alt="Suspected Area overlay" style={{ width: "100%", borderRadius: 6 }} /> : <div>No overlay returned</div>}
        </div>
        <div className="panel">
          <div className="panel__title">Mask</div>
          {maskSrc ? <img src={maskSrc} alt="Mask" style={{ width: "100%", borderRadius: 6 }} /> : <div>No mask returned</div>}
        </div>
        <div className="panel">
          <div className="panel__title">Heatmap Overlay</div>
          {heatmapSrc ? <img src={heatmapSrc} alt="Heatmap Overlay" style={{ width: "100%", borderRadius: 6 }} /> : <div>No heatmap returned</div>}
        </div>
      </div>
    </section>
  );
}

import { memo } from "react";
import { CATEGORY_COLORS } from "../model/data.js";

export default memo(function PredictionPanel({ probs, targetWord, inputWords, vocabSize, id2word, wordCategories, maxSpeed }) {
  const sortedIndices = [...Array(vocabSize).keys()].sort(
    (a, b) => probs[b] - probs[a]
  );

  return (
    <div style={{
      background: "rgba(255,255,255,0.03)",
      borderRadius: 16,
      border: "1px solid rgba(255,255,255,0.08)",
      padding: "20px 24px",
      height: "100%",
      display: "flex",
      flexDirection: "column",
    }}>
      <div style={{ marginBottom: 8 }}>
        <div style={{
          fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
          fontSize: 11,
          color: "#64748b",
          textTransform: "uppercase",
          letterSpacing: 2,
          marginBottom: 8,
        }}>
          Prediction Probabilities
        </div>
        <div style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 12,
          color: "#94a3b8",
          marginBottom: 4,
        }}>
          Input: <span style={{ color: "#e2e8f0" }}>"{inputWords.join(" ")}"</span>
          {targetWord && (
            <span>
              {" → "}
              <span style={{ color: "#4ade80" }}>{targetWord}</span>
            </span>
          )}
        </div>
      </div>
      <div style={{ flex: 1, overflow: "auto" }}>
        {sortedIndices.map((idx) => {
          const word = id2word[idx];
          const prob = probs[idx];
          const isTarget = word === targetWord;
          const barColor = isTarget ? "#4ade80" : CATEGORY_COLORS[wordCategories[word]] || "#94a3b8";

          return (
            <div
              key={word}
              style={{
                display: "flex",
                alignItems: "center",
                marginBottom: 5,
                gap: 8,
              }}
            >
              <div style={{
                width: 48,
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 12,
                color: isTarget ? "#4ade80" : "#cbd5e1",
                textAlign: "right",
                fontWeight: isTarget ? 700 : 400,
              }}>
                {word}
              </div>
              <div style={{
                flex: 1,
                height: 20,
                background: "rgba(255,255,255,0.04)",
                borderRadius: 4,
                overflow: "hidden",
                position: "relative",
              }}>
                <div
                  style={{
                    height: "100%",
                    width: `${Math.max(prob * 100, 0.5)}%`,
                    background: `linear-gradient(90deg, ${barColor}88, ${barColor})`,
                    borderRadius: 4,
                    transition: maxSpeed ? "none" : "width 0.3s ease",
                    boxShadow: prob > 0.3 ? `0 0 12px ${barColor}44` : "none",
                  }}
                />
              </div>
              <div style={{
                width: 44,
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 11,
                color: isTarget ? "#4ade80" : "#64748b",
                fontWeight: isTarget ? 700 : 400,
              }}>
                {(prob * 100).toFixed(1)}%
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
});

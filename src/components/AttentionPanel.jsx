import { useState, memo } from "react";

export default memo(function AttentionPanel({ attnWeights, inputWords, numHeads, numBlocks, maxSpeed }) {
  const [headIdx, setHeadIdx] = useState(0);
  const [blockIdx, setBlockIdx] = useState(0);

  if (!attnWeights || attnWeights.length === 0) return null;

  // attnWeights shape: [numBlocks][numHeads][seqLen][seqLen]
  const blockWeights = attnWeights[blockIdx] || [];
  const weights = blockWeights[headIdx] || [];
  const len = inputWords.length;
  const cellSize = Math.min(36, 220 / len);

  const btnStyle = (active) => ({
    background: active ? "#60a5fa" : "rgba(255,255,255,0.06)",
    color: active ? "#0f172a" : "#64748b",
    border: "none",
    borderRadius: 6,
    padding: "3px 10px",
    fontSize: 10,
    fontFamily: "'JetBrains Mono', monospace",
    cursor: "pointer",
    fontWeight: 600,
  });

  return (
    <div style={{
      background: "rgba(255,255,255,0.03)",
      borderRadius: 16,
      border: "1px solid rgba(255,255,255,0.08)",
      padding: "20px 24px",
      height: "100%",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12, flexWrap: "wrap", gap: 8 }}>
        <div style={{
          fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
          fontSize: 11,
          color: "#64748b",
          textTransform: "uppercase",
          letterSpacing: 2,
        }}>
          Attention Heatmap
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {numBlocks > 1 && (
            <div style={{ display: "flex", gap: 4 }}>
              {Array.from({ length: numBlocks }, (_, i) => (
                <button key={i} onClick={() => setBlockIdx(i)} style={btnStyle(blockIdx === i)}>
                  L{i + 1}
                </button>
              ))}
            </div>
          )}
          <div style={{ display: "flex", gap: 4 }}>
            {Array.from({ length: numHeads }, (_, i) => (
              <button key={i} onClick={() => setHeadIdx(i)} style={btnStyle(headIdx === i)}>
                H{i + 1}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div style={{ display: "flex", justifyContent: "center" }}>
        <div>
          <div style={{ display: "flex", marginLeft: cellSize + 4 }}>
            {inputWords.map((w, j) => (
              <div
                key={j}
                style={{
                  width: cellSize,
                  height: 28,
                  display: "flex",
                  alignItems: "flex-end",
                  justifyContent: "center",
                  fontSize: 9,
                  fontFamily: "'JetBrains Mono', monospace",
                  color: "#94a3b8",
                  transform: "rotate(-35deg)",
                  transformOrigin: "bottom center",
                }}
              >
                {w}
              </div>
            ))}
          </div>
          {inputWords.map((w, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center" }}>
              <div style={{
                width: cellSize,
                fontSize: 9,
                fontFamily: "'JetBrains Mono', monospace",
                color: "#94a3b8",
                textAlign: "right",
                paddingRight: 4,
                overflow: "hidden",
              }}>
                {w}
              </div>
              {inputWords.map((_, j) => {
                const val = weights[i] ? weights[i][j] || 0 : 0;
                const masked = j > i;
                return (
                  <div
                    key={j}
                    style={{
                      width: cellSize - 2,
                      height: cellSize - 2,
                      margin: 1,
                      borderRadius: 3,
                      background: masked
                        ? "rgba(255,255,255,0.02)"
                        : `rgba(96, 165, 250, ${val * 0.9 + 0.05})`,
                      transition: maxSpeed ? "none" : "background 0.3s ease",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 8,
                      fontFamily: "'JetBrains Mono', monospace",
                      color: masked ? "transparent" : val > 0.3 ? "#0f172a" : "rgba(255,255,255,0.3)",
                      fontWeight: 600,
                    }}
                    title={`${inputWords[i]} → ${inputWords[j]}: ${(val * 100).toFixed(1)}%`}
                  >
                    {!masked && val > 0.08 ? (val * 100).toFixed(0) : ""}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
      <div style={{
        textAlign: "center", marginTop: 8,
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 9, color: "#475569"
      }}>
        ← Keys (attending to) | Queries (from) ↓
      </div>
    </div>
  );
});

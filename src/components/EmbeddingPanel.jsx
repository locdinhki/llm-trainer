import { useMemo, memo } from "react";
import { pca2D } from "../model/math.js";
import { CATEGORY_COLORS } from "../model/data.js";

export default memo(function EmbeddingPanel({ embeddings, id2word, wordCategories }) {
  const points = useMemo(() => pca2D(embeddings), [embeddings]);

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;
  const padding = 50;
  const w = 400, h = 280;

  // Only show categories that exist in the current vocabulary
  const activeCategories = [...new Set(Object.values(wordCategories))];

  return (
    <div style={{
      background: "rgba(255,255,255,0.03)",
      borderRadius: 16,
      border: "1px solid rgba(255,255,255,0.08)",
      padding: "20px 24px",
      height: "100%",
    }}>
      <div style={{
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        fontSize: 11,
        color: "#64748b",
        textTransform: "uppercase",
        letterSpacing: 2,
        marginBottom: 8,
      }}>
        Embedding Space (PCA)
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} style={{ width: "100%", height: "auto" }}>
        <line x1={w / 2} y1={padding / 2} x2={w / 2} y2={h - padding / 2} stroke="rgba(255,255,255,0.05)" strokeDasharray="4 4" />
        <line x1={padding / 2} y1={h / 2} x2={w - padding / 2} y2={h / 2} stroke="rgba(255,255,255,0.05)" strokeDasharray="4 4" />

        {points.map((p, i) => {
          const cx = padding + ((p.x - xMin) / xRange) * (w - 2 * padding);
          const cy = padding + ((p.y - yMin) / yRange) * (h - 2 * padding);
          const word = id2word[i];
          const cat = wordCategories[word];
          const color = CATEGORY_COLORS[cat] || "#94a3b8";

          return (
            <g key={word}>
              <circle cx={cx} cy={cy} r={12} fill={color} opacity={0.12} />
              <circle cx={cx} cy={cy} r={5} fill={color} stroke={color} strokeWidth={1} opacity={0.9} />
              <text
                x={cx}
                y={cy - 10}
                textAnchor="middle"
                fill={color}
                fontSize={10}
                fontFamily="'JetBrains Mono', monospace"
                fontWeight={600}
              >
                {word}
              </text>
            </g>
          );
        })}
      </svg>
      <div style={{ display: "flex", gap: 16, justifyContent: "center", marginTop: 4, flexWrap: "wrap" }}>
        {activeCategories.map((cat) => (
          <div key={cat} style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: CATEGORY_COLORS[cat] || "#94a3b8" }} />
            <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#64748b" }}>{cat}</span>
          </div>
        ))}
      </div>
    </div>
  );
});

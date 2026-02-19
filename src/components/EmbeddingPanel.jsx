import { useMemo, useState, memo } from "react";
import { pca2D } from "../model/math.js";
import { CATEGORY_COLORS } from "../model/data.js";

const LABEL_THRESHOLD = 30;

export default memo(function EmbeddingPanel({ embeddings, id2word, wordCategories }) {
  const [hoveredIdx, setHoveredIdx] = useState(null);
  const [activeCategory, setActiveCategory] = useState(null);

  const points = useMemo(() => pca2D(embeddings), [embeddings]);
  const vocabSize = id2word.length;
  const alwaysShowLabels = vocabSize <= LABEL_THRESHOLD;

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;
  const padding = 50;
  const w = 400, h = 280;

  const activeCategories = [...new Set(Object.values(wordCategories))];

  const opacityForIdx = (i) => {
    if (vocabSize <= 1) return 0.9;
    return 0.3 + 0.7 * (1 - i / vocabSize);
  };

  const hoveredCat = hoveredIdx !== null ? wordCategories[id2word[hoveredIdx]] : null;

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
          const isSubword = cat === "subword";
          const isHovered = hoveredIdx === i;
          const sameCategory = hoveredIdx !== null && hoveredCat === cat;
          const isFiltered = activeCategory !== null && cat !== activeCategory;
          const baseOpacity = isFiltered ? 0.08 : opacityForIdx(i);
          const finalOpacity = isHovered ? 1 : (sameCategory && !isFiltered ? 0.85 : baseOpacity);
          const showLabel = alwaysShowLabels || isHovered;
          const r = isSubword ? 4 : 5;

          return (
            <g
              key={i}
              style={{ cursor: "pointer" }}
              onMouseEnter={() => setHoveredIdx(i)}
              onMouseLeave={() => setHoveredIdx(null)}
            >
              {/* Highlight halo */}
              {(isHovered || sameCategory) && !isFiltered && (
                <circle cx={cx} cy={cy} r={14} fill={color} opacity={0.08} />
              )}
              {/* Shape: diamond for subword, circle otherwise */}
              {isSubword ? (
                <polygon
                  points={`${cx},${cy - r - 1} ${cx + r},${cy} ${cx},${cy + r + 1} ${cx - r},${cy}`}
                  fill={color}
                  opacity={finalOpacity}
                  stroke={isHovered ? color : "none"}
                  strokeWidth={1}
                />
              ) : (
                <>
                  <circle cx={cx} cy={cy} r={12} fill={color} opacity={isFiltered ? 0.03 : 0.10} />
                  <circle cx={cx} cy={cy} r={r} fill={color} stroke={color} strokeWidth={1} opacity={finalOpacity} />
                </>
              )}
              {/* Label */}
              {showLabel && (
                <text
                  x={cx}
                  y={cy - 10}
                  textAnchor="middle"
                  fill={color}
                  fontSize={alwaysShowLabels ? 10 : 11}
                  fontFamily="'JetBrains Mono', monospace"
                  fontWeight={600}
                  style={{ pointerEvents: "none", userSelect: "none" }}
                >
                  {word}
                </text>
              )}
            </g>
          );
        })}

        {/* Tooltip */}
        {hoveredIdx !== null && (() => {
          const i = hoveredIdx;
          const p = points[i];
          const cx = padding + ((p.x - xMin) / xRange) * (w - 2 * padding);
          const cy = padding + ((p.y - yMin) / yRange) * (h - 2 * padding);
          const word = id2word[i];
          const cat = wordCategories[word];
          const color = CATEGORY_COLORS[cat] || "#94a3b8";
          const emb = embeddings[i];
          const norm = emb ? Math.sqrt(emb.reduce((s, v) => s + v * v, 0)).toFixed(2) : "—";
          const ttx = cx > w * 0.6 ? cx - 112 : cx + 12;
          const tty = cy > h * 0.6 ? cy - 52 : cy + 4;
          return (
            <g style={{ pointerEvents: "none" }}>
              <rect x={ttx} y={tty} width={100} height={44}
                rx={4}
                fill="#0f172a"
                stroke={color}
                strokeWidth={0.8}
                opacity={0.95}
              />
              <text x={ttx + 6} y={tty + 14} fill={color} fontSize={10} fontFamily="'JetBrains Mono', monospace" fontWeight={700}>{word}</text>
              <text x={ttx + 6} y={tty + 26} fill="#64748b" fontSize={9} fontFamily="'JetBrains Mono', monospace">{cat}</text>
              <text x={ttx + 6} y={tty + 38} fill="#475569" fontSize={9} fontFamily="'JetBrains Mono', monospace">norm={norm}</text>
            </g>
          );
        })()}
      </svg>

      {/* Category filter buttons */}
      <div style={{ display: "flex", gap: 8, justifyContent: "center", marginTop: 4, flexWrap: "wrap" }}>
        <button
          onClick={() => setActiveCategory(null)}
          style={{
            background: activeCategory === null ? "rgba(255,255,255,0.12)" : "rgba(255,255,255,0.04)",
            border: `1px solid ${activeCategory === null ? "rgba(255,255,255,0.2)" : "rgba(255,255,255,0.06)"}`,
            borderRadius: 6,
            padding: "3px 10px",
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 9,
            color: activeCategory === null ? "#e2e8f0" : "#64748b",
            cursor: "pointer",
          }}
        >
          All
        </button>
        {activeCategories.map((cat) => {
          const catColor = CATEGORY_COLORS[cat] || "#94a3b8";
          const isActive = activeCategory === cat;
          return (
            <button
              key={cat}
              onClick={() => setActiveCategory(isActive ? null : cat)}
              style={{
                background: isActive ? `${catColor}22` : "rgba(255,255,255,0.02)",
                border: `1px solid ${isActive ? catColor + "66" : "rgba(255,255,255,0.06)"}`,
                borderRadius: 6,
                padding: "3px 10px",
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 9,
                color: isActive ? catColor : "#64748b",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: 4,
              }}
            >
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: catColor, display: "inline-block" }} />
              {cat}
            </button>
          );
        })}
      </div>
    </div>
  );
});

import { useState, useRef, useEffect, useCallback, memo } from "react";

function truncToken(s, max) {
  return s.length > max ? s.slice(0, max - 1) + "\u2026" : s;
}

function drawDetailHeatmap(ctx, canvasW, canvasH, attnWeights, blockIdx, headIdx, tokens) {
  const n = tokens.length;
  if (n === 0) return;
  const LABEL_W = 52;
  const LABEL_H = 40;
  const cellW = (canvasW - LABEL_W) / n;
  const cellH = (canvasH - LABEL_H) / n;

  ctx.clearRect(0, 0, canvasW, canvasH);

  const weights = (attnWeights[blockIdx] || [])[headIdx] || [];

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const masked = j > i;
      const val = weights[i] ? (weights[i][j] || 0) : 0;
      const x = LABEL_W + j * cellW;
      const y = LABEL_H + i * cellH;

      if (masked) {
        ctx.fillStyle = "rgba(255,255,255,0.02)";
      } else {
        const a = val * 0.9 + 0.05;
        ctx.fillStyle = `rgba(96,165,250,${a.toFixed(3)})`;
      }
      ctx.fillRect(x + 0.5, y + 0.5, cellW - 1, cellH - 1);

      if (!masked && val > 0.08 && cellW > 18) {
        ctx.fillStyle = val > 0.3 ? "#0f172a" : "rgba(255,255,255,0.5)";
        ctx.font = `bold ${Math.max(8, Math.min(10, cellW * 0.4))}px 'JetBrains Mono', monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(Math.round(val * 100).toString(), x + cellW / 2, y + cellH / 2);
      }
    }

    // Row label
    const label = truncToken(tokens[i], 6);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(label, LABEL_W - 4, LABEL_H + i * cellH + cellH / 2);
  }

  // Column labels (rotated)
  for (let j = 0; j < n; j++) {
    const label = truncToken(tokens[j], 6);
    const x = LABEL_W + j * cellW + cellW / 2;
    ctx.save();
    ctx.translate(x, LABEL_H - 4);
    ctx.rotate(-Math.PI / 5);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(label, 0, 0);
    ctx.restore();
  }
}

function drawOverviewGrid(ctx, canvasW, canvasH, attnWeights, numBlocks, numHeads, seqLen) {
  ctx.clearRect(0, 0, canvasW, canvasH);

  const GAP = 6;
  const LABEL_H = 16;
  const LABEL_W = 20;
  const thumbW = Math.floor((canvasW - LABEL_W - GAP * (numHeads + 1)) / numHeads);
  const thumbH = Math.floor((canvasH - LABEL_H - GAP * (numBlocks + 1)) / numBlocks);

  for (let b = 0; b < numBlocks; b++) {
    for (let h = 0; h < numHeads; h++) {
      const x0 = LABEL_W + GAP + h * (thumbW + GAP);
      const y0 = LABEL_H + GAP + b * (thumbH + GAP);
      const weights = (attnWeights[b] || [])[h] || [];
      const cellW = thumbW / seqLen;
      const cellH = thumbH / seqLen;

      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < seqLen; j++) {
          const masked = j > i;
          const val = weights[i] ? (weights[i][j] || 0) : 0;
          if (masked) {
            ctx.fillStyle = "rgba(255,255,255,0.03)";
          } else {
            const a = val * 0.9 + 0.05;
            ctx.fillStyle = `rgba(96,165,250,${a.toFixed(3)})`;
          }
          ctx.fillRect(x0 + j * cellW, y0 + i * cellH, Math.ceil(cellW), Math.ceil(cellH));
        }
      }

      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.lineWidth = 0.5;
      ctx.strokeRect(x0, y0, thumbW, thumbH);
    }
  }

  // Head labels along top
  ctx.fillStyle = "#475569";
  ctx.font = "9px 'JetBrains Mono', monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let h = 0; h < numHeads; h++) {
    const x0 = LABEL_W + GAP + h * (thumbW + GAP) + thumbW / 2;
    ctx.fillText(`H${h + 1}`, x0, 2);
  }

  // Block labels along left
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (let b = 0; b < numBlocks; b++) {
    const y0 = LABEL_H + GAP + b * (thumbH + GAP) + thumbH / 2;
    ctx.fillText(`L${b + 1}`, LABEL_W / 2, y0);
  }
}

export default memo(function AttentionPanel({ attnWeights, inputWords, numHeads, numBlocks, maxSpeed }) {
  const [headIdx, setHeadIdx] = useState(0);
  const [blockIdx, setBlockIdx] = useState(0);
  const [viewMode, setViewMode] = useState("overview");

  const detailCanvasRef = useRef(null);
  const overviewCanvasRef = useRef(null);

  const seqLen = inputWords ? inputWords.length : 0;
  const useCanvasMode = seqLen > 16;

  // Canvas draw effect — always called, guards internally
  useEffect(() => {
    if (!useCanvasMode || !attnWeights || attnWeights.length === 0) return;
    if (viewMode === "detail") {
      const canvas = detailCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      drawDetailHeatmap(ctx, canvas.width, canvas.height, attnWeights, blockIdx, headIdx, inputWords);
    } else {
      const canvas = overviewCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      drawOverviewGrid(ctx, canvas.width, canvas.height, attnWeights, numBlocks, numHeads, seqLen);
    }
  }, [useCanvasMode, attnWeights, blockIdx, headIdx, viewMode, numBlocks, numHeads, seqLen, inputWords]);

  const handleOverviewClick = useCallback((e) => {
    const canvas = overviewCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    const GAP = 6;
    const LABEL_H = 16;
    const LABEL_W = 20;
    const thumbW = Math.floor((canvas.width - LABEL_W - GAP * (numHeads + 1)) / numHeads);
    const thumbH = Math.floor((canvas.height - LABEL_H - GAP * (numBlocks + 1)) / numBlocks);
    const h = Math.floor((x - LABEL_W - GAP) / (thumbW + GAP));
    const b = Math.floor((y - LABEL_H - GAP) / (thumbH + GAP));
    if (h >= 0 && h < numHeads && b >= 0 && b < numBlocks) {
      setBlockIdx(b);
      setHeadIdx(h);
      setViewMode("detail");
    }
  }, [numHeads, numBlocks]);

  if (!attnWeights || attnWeights.length === 0) return null;

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

  // Canvas mode (seqLen > 16)
  if (useCanvasMode) {
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
            {viewMode === "detail" && (
              <>
                <button onClick={() => setViewMode("overview")} style={btnStyle(false)}>
                  Overview
                </button>
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
              </>
            )}
            {viewMode === "overview" && (
              <span style={{ fontSize: 9, color: "#475569", fontFamily: "'JetBrains Mono', monospace" }}>
                click to expand
              </span>
            )}
          </div>
        </div>

        {viewMode === "overview" && (
          <canvas
            ref={overviewCanvasRef}
            width={400}
            height={Math.max(120, 16 + (numBlocks * 66))}
            style={{ width: "100%", height: "auto", cursor: "pointer", borderRadius: 6 }}
            onClick={handleOverviewClick}
          />
        )}

        {viewMode === "detail" && (
          <>
            <canvas
              ref={detailCanvasRef}
              width={400}
              height={380}
              style={{ width: "100%", height: "auto", borderRadius: 6 }}
            />
            <div style={{
              textAlign: "center", marginTop: 6,
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 9, color: "#475569",
            }}>
              Block {blockIdx + 1} / Head {headIdx + 1} — {seqLen}x{seqLen} attention
            </div>
          </>
        )}
      </div>
    );
  }

  // DOM mode (seqLen <= 16)
  const blockWeights = attnWeights[blockIdx] || [];
  const weights = blockWeights[headIdx] || [];
  const cellSize = Math.min(36, 220 / seqLen);

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

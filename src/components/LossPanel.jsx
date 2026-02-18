import { memo } from "react";

export default memo(function LossPanel({ lossHistory, vocabSize }) {
  if (lossHistory.length === 0) return (
    <div style={{
      background: "rgba(255,255,255,0.03)",
      borderRadius: 16,
      border: "1px solid rgba(255,255,255,0.08)",
      padding: "20px 24px",
      height: "100%",
      display: "flex",
      flexDirection: "column",
    }}>
      <div style={{
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        fontSize: 11,
        color: "#64748b",
        textTransform: "uppercase",
        letterSpacing: 2,
        marginBottom: 8,
      }}>
        Training Loss
      </div>
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "#475569", fontSize: 13, fontFamily: "'JetBrains Mono', monospace" }}>
        Press Step or Play to begin training
      </div>
    </div>
  );

  const w = 400, h = 200;
  const pad = { top: 20, right: 20, bottom: 30, left: 45 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  const maxLoss = Math.max(...lossHistory, 2.5);
  const minLoss = Math.min(...lossHistory, 0);
  const lossRange = maxLoss - minLoss || 1;

  const pathD = lossHistory
    .map((loss, i) => {
      const x = pad.left + (i / Math.max(lossHistory.length - 1, 1)) * plotW;
      const y = pad.top + (1 - (loss - minLoss) / lossRange) * plotH;
      return `${i === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");

  const randomLoss = -Math.log(1 / vocabSize);
  const randomY = pad.top + (1 - (randomLoss - minLoss) / lossRange) * plotH;

  return (
    <div style={{
      background: "rgba(255,255,255,0.03)",
      borderRadius: 16,
      border: "1px solid rgba(255,255,255,0.08)",
      padding: "20px 24px",
      height: "100%",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <div style={{
          fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
          fontSize: 11,
          color: "#64748b",
          textTransform: "uppercase",
          letterSpacing: 2,
        }}>
          Training Loss
        </div>
        <div style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 12,
          color: "#f97316",
        }}>
          {lossHistory[lossHistory.length - 1]?.toFixed(4)}
        </div>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} style={{ width: "100%", height: "auto" }}>
        {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
          const y = pad.top + frac * plotH;
          const val = maxLoss - frac * lossRange;
          return (
            <g key={frac}>
              <line x1={pad.left} y1={y} x2={w - pad.right} y2={y} stroke="rgba(255,255,255,0.04)" />
              <text x={pad.left - 6} y={y + 3} textAnchor="end" fill="#475569" fontSize={9} fontFamily="'JetBrains Mono', monospace">
                {val.toFixed(1)}
              </text>
            </g>
          );
        })}

        {randomY > pad.top && randomY < pad.top + plotH && (
          <>
            <line x1={pad.left} y1={randomY} x2={w - pad.right} y2={randomY} stroke="#f9731644" strokeDasharray="6 4" />
            <text x={w - pad.right - 2} y={randomY - 5} textAnchor="end" fill="#f97316" fontSize={8} fontFamily="'JetBrains Mono', monospace" opacity={0.6}>
              random guess
            </text>
          </>
        )}

        <path d={pathD} fill="none" stroke="#f97316" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
        <path d={pathD} fill="none" stroke="#f97316" strokeWidth={6} strokeLinecap="round" opacity={0.15} />

        {lossHistory.length > 0 && (
          <circle
            cx={pad.left + ((lossHistory.length - 1) / Math.max(lossHistory.length - 1, 1)) * plotW}
            cy={pad.top + (1 - (lossHistory[lossHistory.length - 1] - minLoss) / lossRange) * plotH}
            r={4}
            fill="#f97316"
          />
        )}

        <text x={pad.left + plotW / 2} y={h - 4} textAnchor="middle" fill="#475569" fontSize={9} fontFamily="'JetBrains Mono', monospace">
          Training Steps
        </text>
      </svg>
    </div>
  );
});

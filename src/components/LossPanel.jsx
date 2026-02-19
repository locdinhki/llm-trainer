import { memo } from "react";

function downsample(arr, maxPts) {
  if (arr.length <= maxPts) return arr;
  const step = arr.length / maxPts;
  return Array.from({ length: maxPts }, (_, i) => arr[Math.floor(i * step)]);
}

const MAX_DISPLAY = 400;
const EMA_ALPHA = 0.1;

export default memo(function LossPanel({
  lossHistory,
  vocabSize,
  lrHistory = [],
  useLRSchedule = false,
  trainingDataSize = 0,
  batchSize = 1,
}) {
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

  // EMA smoothed loss
  const emaHistory = [];
  let ema = lossHistory[0];
  for (let i = 0; i < lossHistory.length; i++) {
    ema = EMA_ALPHA * lossHistory[i] + (1 - EMA_ALPHA) * ema;
    emaHistory.push(ema);
  }

  // Downsample for display
  const displayRaw = downsample(lossHistory, MAX_DISPLAY);
  const displayEMA = downsample(emaHistory, MAX_DISPLAY);

  const w = 400, h = 200;
  const pad = { top: 20, right: 20, bottom: 30, left: 45 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  const allVals = displayRaw.concat(displayEMA);
  const maxLoss = Math.max(...allVals, 2.5);
  const minLoss = Math.min(...allVals, 0);
  const lossRange = maxLoss - minLoss || 1;

  const toX = (i, len) => pad.left + (i / Math.max(len - 1, 1)) * plotW;
  const toY = (val) => pad.top + (1 - (val - minLoss) / lossRange) * plotH;

  const rawPathD = displayRaw
    .map((loss, i) => `${i === 0 ? "M" : "L"} ${toX(i, displayRaw.length)} ${toY(loss)}`)
    .join(" ");

  const emaPathD = displayEMA
    .map((loss, i) => `${i === 0 ? "M" : "L"} ${toX(i, displayEMA.length)} ${toY(loss)}`)
    .join(" ");

  const randomLoss = -Math.log(1 / vocabSize);
  const randomY = toY(randomLoss);

  // Epoch markers
  const stepsPerEpoch = trainingDataSize > 0 && batchSize > 0
    ? Math.max(1, Math.ceil(trainingDataSize / batchSize))
    : 0;

  // Current values
  const currentLoss = lossHistory[lossHistory.length - 1];
  const currentPerplexity = Math.exp(currentLoss);
  const currentLR = lrHistory.length > 0 ? lrHistory[lrHistory.length - 1] : null;
  const currentEMA = emaHistory[emaHistory.length - 1];

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
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          {useLRSchedule && currentLR !== null && (
            <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#60a5fa" }}>
              lr={currentLR.toExponential(2)}
            </span>
          )}
          <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#a78bfa" }}>
            ppl={currentPerplexity.toFixed(1)}
          </span>
          <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#f97316" }}>
            {currentLoss.toFixed(4)}
          </span>
        </div>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} style={{ width: "100%", height: "auto" }}>
        {/* Y-axis grid lines */}
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

        {/* Epoch markers */}
        {stepsPerEpoch > 0 && Array.from(
          { length: Math.floor(lossHistory.length / stepsPerEpoch) },
          (_, i) => {
            const epochStep = (i + 1) * stepsPerEpoch;
            if (epochStep >= lossHistory.length) return null;
            const x = pad.left + (epochStep / Math.max(lossHistory.length - 1, 1)) * plotW;
            // Skip markers too close together (< 12px apart)
            const prevX = i > 0 ? pad.left + (i * stepsPerEpoch / Math.max(lossHistory.length - 1, 1)) * plotW : 0;
            if (i > 0 && x - prevX < 12) return null;
            return (
              <line
                key={`epoch-${i}`}
                x1={x} y1={pad.top}
                x2={x} y2={pad.top + plotH}
                stroke="rgba(167,139,250,0.2)"
                strokeDasharray="3 3"
              />
            );
          }
        )}

        {/* Random guess baseline */}
        {randomY > pad.top && randomY < pad.top + plotH && (
          <>
            <line x1={pad.left} y1={randomY} x2={w - pad.right} y2={randomY} stroke="#f9731644" strokeDasharray="6 4" />
            <text x={w - pad.right - 2} y={randomY - 5} textAnchor="end" fill="#f97316" fontSize={8} fontFamily="'JetBrains Mono', monospace" opacity={0.6}>
              random guess
            </text>
          </>
        )}

        {/* Raw loss (faded thin line) */}
        <path d={rawPathD} fill="none" stroke="#f97316" strokeWidth={1} strokeLinecap="round" strokeLinejoin="round" opacity={0.25} />

        {/* EMA smoothed loss (solid thick line) */}
        <path d={emaPathD} fill="none" stroke="#f97316" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round" />
        <path d={emaPathD} fill="none" stroke="#f97316" strokeWidth={6} strokeLinecap="round" opacity={0.15} />

        {/* Endpoint dot on EMA line */}
        {displayEMA.length > 0 && (
          <circle
            cx={toX(displayEMA.length - 1, displayEMA.length)}
            cy={toY(currentEMA)}
            r={4}
            fill="#f97316"
          />
        )}

        <text x={pad.left + plotW / 2} y={h - 4} textAnchor="middle" fill="#475569" fontSize={9} fontFamily="'JetBrains Mono', monospace">
          Training Steps ({lossHistory.length})
        </text>
      </svg>
    </div>
  );
});

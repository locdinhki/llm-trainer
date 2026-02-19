import { memo, useMemo } from "react";
import { getLearningRate } from "../model/training.js";

const headerStyle = {
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 10,
  color: "#64748b",
  textTransform: "uppercase",
  letterSpacing: 1.5,
};

const valueStyle = {
  fontFamily: "'JetBrains Mono', monospace",
  fontSize: 11,
};

function MiniChart({ title, data, color, referenceValue, referenceLabel, referenceColor, valueFormat }) {
  const w = 300, h = 140;
  const pad = { top: 16, right: 12, bottom: 20, left: 40 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  if (data.length === 0) {
    return (
      <div style={{ background: "rgba(255,255,255,0.02)", borderRadius: 12, padding: "12px 14px", border: "1px solid rgba(255,255,255,0.05)" }}>
        <div style={headerStyle}>{title}</div>
        <div style={{ height: 80, display: "flex", alignItems: "center", justifyContent: "center", color: "#475569", fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}>
          —
        </div>
      </div>
    );
  }

  const allVals = referenceValue != null ? [...data, referenceValue] : data;
  const maxVal = Math.max(...allVals);
  const minVal = Math.min(...allVals, 0);
  const range = maxVal - minVal || 1;

  const toY = (v) => pad.top + (1 - (v - minVal) / range) * plotH;
  const toX = (i) => pad.left + (i / Math.max(data.length - 1, 1)) * plotW;

  const pathD = data
    .map((v, i) => `${i === 0 ? "M" : "L"} ${toX(i)} ${toY(v)}`)
    .join(" ");

  const lastVal = data[data.length - 1];
  const refY = referenceValue != null ? toY(referenceValue) : null;
  const refVisible = refY != null && refY > pad.top - 5 && refY < pad.top + plotH + 5;

  // Y-axis ticks
  const ticks = [0, 0.5, 1].map((frac) => ({
    y: pad.top + frac * plotH,
    val: maxVal - frac * range,
  }));

  return (
    <div style={{ background: "rgba(255,255,255,0.02)", borderRadius: 12, padding: "12px 14px", border: "1px solid rgba(255,255,255,0.05)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 2 }}>
        <div style={headerStyle}>{title}</div>
        <div style={{ ...valueStyle, color }}>{valueFormat(lastVal)}</div>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} style={{ width: "100%", height: "auto" }}>
        {ticks.map(({ y, val }, i) => (
          <g key={i}>
            <line x1={pad.left} y1={y} x2={w - pad.right} y2={y} stroke="rgba(255,255,255,0.04)" />
            <text x={pad.left - 4} y={y + 3} textAnchor="end" fill="#475569" fontSize={8} fontFamily="'JetBrains Mono', monospace">
              {val < 0.01 && val > 0 ? val.toExponential(0) : val < 10 ? val.toFixed(1) : Math.round(val)}
            </text>
          </g>
        ))}

        {refVisible && (
          <>
            <line x1={pad.left} y1={refY} x2={w - pad.right} y2={refY} stroke={referenceColor || "#f9731644"} strokeDasharray="4 3" />
            <text x={w - pad.right - 2} y={refY - 4} textAnchor="end" fill={referenceColor || "#f97316"} fontSize={7} fontFamily="'JetBrains Mono', monospace" opacity={0.6}>
              {referenceLabel}
            </text>
          </>
        )}

        <path d={pathD} fill="none" stroke={color} strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
        <path d={pathD} fill="none" stroke={color} strokeWidth={5} strokeLinecap="round" opacity={0.1} />

        <circle cx={toX(data.length - 1)} cy={toY(lastVal)} r={3} fill={color} />
      </svg>
    </div>
  );
}

export default memo(function MetricsPanel({
  lossHistory,
  gradNormHistory,
  lrHistory,
  vocabSize,
  currentStep,
  warmupSteps,
  totalSteps,
  baseLR,
  useLRSchedule,
}) {
  const perplexityHistory = useMemo(
    () => lossHistory.map((l) => Math.exp(l)),
    [lossHistory]
  );

  // Precompute theoretical LR curve for overlay
  const theoreticalLR = useMemo(() => {
    if (!useLRSchedule) return [];
    const points = [];
    const numPoints = Math.min(200, totalSteps);
    for (let i = 0; i <= numPoints; i++) {
      const s = Math.round((i / numPoints) * totalSteps);
      points.push(getLearningRate(s, warmupSteps, totalSteps, baseLR));
    }
    return points;
  }, [useLRSchedule, warmupSteps, totalSteps, baseLR]);

  // Merge theoretical + actual for LR chart
  const lrChartData = lrHistory.length > 0 ? lrHistory : theoreticalLR;

  return (
    <div style={{
      gridColumn: "1 / -1",
      background: "rgba(255,255,255,0.03)",
      borderRadius: 16,
      border: "1px solid rgba(255,255,255,0.08)",
      padding: "16px 20px",
    }}>
      <div style={{
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        fontSize: 11,
        color: "#64748b",
        textTransform: "uppercase",
        letterSpacing: 2,
        marginBottom: 12,
      }}>
        Training Metrics
      </div>
      <div style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr 1fr",
        gap: 12,
      }}>
        <MiniChart
          title="Perplexity"
          data={perplexityHistory}
          color="#a78bfa"
          referenceValue={vocabSize}
          referenceLabel="random"
          referenceColor="#a78bfa66"
          valueFormat={(v) => v.toFixed(1)}
        />
        <MiniChart
          title="Gradient Norm"
          data={gradNormHistory}
          color="#22d3ee"
          referenceValue={5.0}
          referenceLabel="clip=5.0"
          referenceColor="#ef444466"
          valueFormat={(v) => v.toFixed(2)}
        />
        <MiniChart
          title="Learning Rate"
          data={lrChartData}
          color="#60a5fa"
          referenceValue={null}
          referenceLabel=""
          valueFormat={(v) => v < 0.001 ? v.toExponential(1) : v.toFixed(4)}
        />
      </div>
    </div>
  );
});

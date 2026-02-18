import { memo } from "react";

export default memo(function SegmentedControl({ options, value, onChange, formatLabel }) {
  return (
    <div style={{ display: "flex", gap: 4 }}>
      {options.map((opt) => (
        <button
          key={opt}
          onClick={() => onChange(opt)}
          style={{
            flex: 1,
            padding: "6px 0",
            borderRadius: 6,
            border: "none",
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 12,
            fontWeight: 600,
            cursor: "pointer",
            background: value === opt ? "#6366f1" : "rgba(255,255,255,0.06)",
            color: value === opt ? "white" : "#64748b",
          }}
        >
          {formatLabel ? formatLabel(opt) : opt}
        </button>
      ))}
    </div>
  );
});

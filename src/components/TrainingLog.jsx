import { useRef, useEffect, useState, memo } from "react";

const TYPE_COLORS = {
  info: "#64748b",
  success: "#4ade80",
  warning: "#fb923c",
  config: "#a78bfa",
};

export default memo(function TrainingLog({ logs }) {
  const scrollRef = useRef(null);
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    if (scrollRef.current && !collapsed) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, collapsed]);

  return (
    <div style={{
      background: "rgba(255,255,255,0.03)",
      borderRadius: 16,
      border: "1px solid rgba(255,255,255,0.08)",
      gridColumn: "1 / -1",
      overflow: "hidden",
    }}>
      <div
        onClick={() => setCollapsed(!collapsed)}
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "10px 20px",
          cursor: "pointer",
          borderBottom: collapsed ? "none" : "1px solid rgba(255,255,255,0.06)",
        }}
      >
        <div style={{
          fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
          fontSize: 11,
          color: "#64748b",
          textTransform: "uppercase",
          letterSpacing: 2,
        }}>
          Training Log
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 10,
            color: "#475569",
          }}>
            {logs.length} entries
          </span>
          <span style={{ color: "#64748b", fontSize: 12 }}>
            {collapsed ? "▸" : "▾"}
          </span>
        </div>
      </div>
      {!collapsed && (
        <div
          ref={scrollRef}
          style={{
            height: 150,
            overflowY: "auto",
            padding: "8px 20px",
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 11,
            lineHeight: 1.8,
          }}
        >
          {logs.length === 0 ? (
            <div style={{ color: "#475569", padding: "20px 0", textAlign: "center" }}>
              Waiting for training to start...
            </div>
          ) : (
            logs.map((log, i) => (
              <div key={i} style={{ color: TYPE_COLORS[log.type] || "#64748b" }}>
                {log.message}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
});

import SegmentedControl from "./controls/SegmentedControl.jsx";

function SectionLabel({ text }) {
  return (
    <div style={{
      fontSize: 10, color: "#64748b", textTransform: "uppercase",
      letterSpacing: 2, marginTop: 20, marginBottom: 10,
      borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: 6,
      fontFamily: "'JetBrains Mono', monospace",
    }}>
      {text}
    </div>
  );
}

function SettingRow({ label, children }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 6, fontFamily: "'JetBrains Mono', monospace" }}>
        {label}
      </div>
      {children}
    </div>
  );
}

export default function SettingsPanel({
  config, onConfigChange,
  learningRate, onLearningRateChange,
  sentencesInput, onSentencesInputChange,
  onApplySentences,
  paramCount,
  onClose,
}) {
  const embedDimOptions = [8, 16, 32];
  const numHeadsOptions = [1, 2, 4].filter((h) => config.embedDim % h === 0);
  const numBlocksOptions = [1, 2, 3, 4];
  const ffnDimOptions = [1, 2, 3, 4].map((m) => m * config.embedDim);

  return (
    <>
      <div onClick={onClose} style={{
        position: "fixed", inset: 0,
        background: "rgba(0,0,0,0.5)",
        zIndex: 99,
      }} />
      <div style={{
        position: "fixed", top: 0, left: 0, bottom: 0,
        width: 360,
        background: "#0f1525",
        borderRight: "1px solid rgba(255,255,255,0.08)",
        zIndex: 100,
        padding: 24,
        overflowY: "auto",
        fontFamily: "'JetBrains Mono', monospace",
        color: "#e2e8f0",
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
          <h2 style={{ fontSize: 16, color: "#e2e8f0", margin: 0, fontFamily: "'JetBrains Mono', monospace" }}>Settings</h2>
          <button onClick={onClose} style={{
            background: "rgba(255,255,255,0.06)",
            color: "#94a3b8",
            border: "none",
            borderRadius: 6,
            padding: "4px 10px",
            fontSize: 12,
            cursor: "pointer",
            fontFamily: "'JetBrains Mono', monospace",
          }}>
            ✕
          </button>
        </div>

        <SectionLabel text="Model Architecture" />

        <SettingRow label="Embedding Dim">
          <SegmentedControl
            options={embedDimOptions}
            value={config.embedDim}
            onChange={(v) => {
              let newNumHeads = config.numHeads;
              if (v % newNumHeads !== 0) newNumHeads = 1;
              let newFfnDim = config.ffnDim;
              if (newFfnDim < v || newFfnDim > v * 4) newFfnDim = v * 2;
              onConfigChange({ ...config, embedDim: v, numHeads: newNumHeads, ffnDim: newFfnDim });
            }}
          />
        </SettingRow>

        <SettingRow label="Attention Heads">
          <SegmentedControl
            options={numHeadsOptions}
            value={config.numHeads}
            onChange={(v) => onConfigChange({ ...config, numHeads: v })}
          />
        </SettingRow>

        <SettingRow label="FFN Hidden Dim">
          <SegmentedControl
            options={ffnDimOptions}
            value={config.ffnDim}
            onChange={(v) => onConfigChange({ ...config, ffnDim: v })}
          />
        </SettingRow>

        <SettingRow label="Transformer Blocks">
          <SegmentedControl
            options={numBlocksOptions}
            value={config.numBlocks}
            onChange={(v) => onConfigChange({ ...config, numBlocks: v })}
          />
        </SettingRow>

        <SettingRow label="Max Sequence Length">
          <input
            type="number"
            min={4}
            max={20}
            value={config.seqLen}
            onChange={(e) => onConfigChange({ ...config, seqLen: Math.max(4, Math.min(20, Number(e.target.value))) })}
            style={{
              width: "100%",
              background: "rgba(255,255,255,0.06)",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 6,
              color: "#e2e8f0",
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 12,
              padding: "6px 10px",
            }}
          />
        </SettingRow>

        <SectionLabel text="Training" />

        <SettingRow label={`Learning Rate: ${learningRate.toFixed(4)}`}>
          <input
            type="range"
            min={0.001}
            max={0.1}
            step={0.001}
            value={learningRate}
            onChange={(e) => onLearningRateChange(Number(e.target.value))}
            style={{ width: "100%", accentColor: "#6366f1" }}
          />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "#475569", marginTop: 2 }}>
            <span>0.001</span><span>0.1</span>
          </div>
        </SettingRow>

        <SectionLabel text="Training Sentences" />

        <textarea
          value={sentencesInput}
          onChange={(e) => onSentencesInputChange(e.target.value)}
          style={{
            width: "100%",
            height: 180,
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: 8,
            color: "#e2e8f0",
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 11,
            padding: 12,
            resize: "vertical",
            lineHeight: 1.6,
          }}
          placeholder="Enter sentences, one per line..."
        />
        <button
          onClick={onApplySentences}
          style={{
            width: "100%",
            marginTop: 8,
            background: "linear-gradient(135deg, #3b82f6, #6366f1)",
            color: "white",
            border: "none",
            borderRadius: 8,
            padding: "10px 0",
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 12,
            fontWeight: 600,
            cursor: "pointer",
          }}
        >
          Apply Sentences & Reset Model
        </button>

        <div style={{ marginTop: 20, padding: 12, background: "rgba(255,255,255,0.03)", borderRadius: 8, border: "1px solid rgba(255,255,255,0.06)" }}>
          <div style={{ fontSize: 10, color: "#64748b", marginBottom: 4 }}>Model Parameters</div>
          <div style={{ fontSize: 18, color: "#a78bfa", fontWeight: 700 }}>{paramCount.toLocaleString()}</div>
        </div>

        <div style={{ marginTop: 12, fontSize: 10, color: "#f97316", opacity: 0.7 }}>
          Changing architecture or sentences resets training progress.
        </div>
      </div>
    </>
  );
}

import { useMemo } from "react";
import SegmentedControl from "./controls/SegmentedControl.jsx";
import { computeCorpusStats } from "../model/data.js";

const PRESETS = [
  { label: "Tiny",   embedDim: 16, numBlocks: 1, numHeads: 2, ffnDim: 32  },
  { label: "Small",  embedDim: 32, numBlocks: 2, numHeads: 4, ffnDim: 96  },
  { label: "Medium", embedDim: 48, numBlocks: 3, numHeads: 4, ffnDim: 128 },
  { label: "Large",  embedDim: 64, numBlocks: 4, numHeads: 8, ffnDim: 256 },
];

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
  activation, onActivationChange,
  dropout, onDropoutChange,
  learningRate, onLearningRateChange,
  batchSize, onBatchSizeChange,
  useAdam, onUseAdamChange,
  useLRSchedule, onUseLRScheduleChange,
  warmupSteps, onWarmupStepsChange,
  totalSteps, onTotalStepsChange,
  corpusPreset, onCorpusChange,
  sentences,
  sentencesInput, onSentencesInputChange,
  onApplySentences,
  weightDecay, onWeightDecayChange,
  tokenizerMode, onTokenizerModeChange,
  bpeVocabSize, onBpeVocabSizeChange,
  tokenizer,
  onSaveCheckpoint, onLoadCheckpoint,
  paramCount,
  onClose,
}) {
  const corpusStats = useMemo(() => computeCorpusStats(sentences), [sentences]);
  const embedDimOptions = [16, 32, 48, 64];
  const numHeadsOptions = [1, 2, 4, 6, 8, 12, 16].filter((h) => config.embedDim % h === 0 && h <= config.embedDim);
  const numBlocksOptions = [1, 2, 3, 4, 6];
  const ffnDimOptions = [2, 3, 4].map((m) => m * config.embedDim);

  // Live parameter count estimate from current settings
  const estimatedParams = useMemo(() => {
    const { embedDim, numBlocks, ffnDim } = config;
    const V = tokenizer.vocabSize;
    const D = embedDim;
    const B = numBlocks;
    const F = ffnDim;
    const embParams = V * D * 2; // embedding + unembedding
    const blockParams = 4 * D * D + 4 * D + 3 * F * D + 3 * F + 2 * D; // attn + FFN + norms
    const outBias = V;
    return embParams + B * blockParams + outBias;
  }, [config, tokenizer.vocabSize]);

  const isLargeConfig = config.embedDim >= 64 && config.numBlocks >= 4;

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

        <SectionLabel text="Presets" />

        <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
          {PRESETS.map(({ label, ...preset }) => {
            const isActive = config.embedDim === preset.embedDim && config.numBlocks === preset.numBlocks
              && config.numHeads === preset.numHeads && config.ffnDim === preset.ffnDim;
            const isLarge = label === "Large";
            return (
              <button
                key={label}
                onClick={() => onConfigChange({ ...config, ...preset })}
                style={{
                  flex: 1,
                  padding: "8px 0",
                  borderRadius: 8,
                  border: isActive
                    ? "1px solid rgba(99, 102, 241, 0.5)"
                    : isLarge
                      ? "1px solid rgba(251, 146, 60, 0.4)"
                      : "1px solid rgba(255,255,255,0.08)",
                  background: isActive
                    ? "rgba(99, 102, 241, 0.15)"
                    : isLarge
                      ? "rgba(251, 146, 60, 0.08)"
                      : "rgba(255,255,255,0.06)",
                  color: isActive ? "#a5b4fc" : isLarge ? "#fb923c" : "#94a3b8",
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 11,
                  fontWeight: 600,
                  cursor: "pointer",
                }}
              >
                {label}
              </button>
            );
          })}
        </div>

        {isLargeConfig && (
          <div style={{
            marginBottom: 12,
            padding: "8px 12px",
            borderRadius: 8,
            background: "rgba(251, 146, 60, 0.08)",
            border: "1px solid rgba(251, 146, 60, 0.25)",
            fontSize: 10,
            color: "#fb923c",
            fontFamily: "'JetBrains Mono', monospace",
            lineHeight: 1.5,
          }}>
            Large config (~{estimatedParams.toLocaleString()} params). Training may be slow in-browser.
          </div>
        )}

        <SectionLabel text="Model Architecture" />

        <SettingRow label="Embedding Dim">
          <SegmentedControl
            options={embedDimOptions}
            value={config.embedDim}
            onChange={(v) => {
              let newNumHeads = config.numHeads;
              if (v % newNumHeads !== 0) newNumHeads = 1;
              let newFfnDim = config.ffnDim;
              if (newFfnDim < v * 2 || newFfnDim > v * 4) newFfnDim = v * 2;
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
          <SegmentedControl
            options={[10, 16, 24, 32, 48]}
            value={config.seqLen}
            onChange={(v) => onConfigChange({ ...config, seqLen: v })}
          />
        </SettingRow>

        <SettingRow label="Activation Function">
          <SegmentedControl
            options={["silu", "gelu", "relu"]}
            value={activation}
            onChange={onActivationChange}
            formatLabel={(v) => v.toUpperCase()}
          />
        </SettingRow>

        <SettingRow label={`Dropout: ${dropout.toFixed(2)}`}>
          <input
            type="range"
            min={0}
            max={0.3}
            step={0.05}
            value={dropout}
            onChange={(e) => onDropoutChange(Number(e.target.value))}
            style={{ width: "100%", accentColor: "#6366f1" }}
          />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "#475569", marginTop: 2 }}>
            <span>0.00</span><span>0.30</span>
          </div>
        </SettingRow>

        <SectionLabel text="Tokenization" />

        <SettingRow label="Mode">
          <SegmentedControl
            options={["Word-level", "BPE"]}
            value={tokenizerMode === "bpe" ? "BPE" : "Word-level"}
            onChange={(v) => onTokenizerModeChange(v === "BPE" ? "bpe" : "word")}
          />
        </SettingRow>

        {tokenizerMode === "bpe" && (
          <SettingRow label={`BPE Vocab Size: ${bpeVocabSize}`}>
            <input
              type="range"
              min={50}
              max={500}
              step={10}
              value={bpeVocabSize}
              onChange={(e) => onBpeVocabSizeChange(Number(e.target.value))}
              style={{ width: "100%", accentColor: "#c084fc" }}
            />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "#475569", marginTop: 2 }}>
              <span>50</span><span>500</span>
            </div>
          </SettingRow>
        )}

        <div style={{
          padding: 10, borderRadius: 8,
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,255,255,0.06)",
          fontSize: 11, color: "#94a3b8",
          lineHeight: 1.6,
        }}>
          <div><span style={{ color: "#64748b" }}>Vocab size:</span> {tokenizer.vocabSize} tokens</div>
          {tokenizerMode === "bpe" && (
            <div><span style={{ color: "#64748b" }}>Merges:</span> {tokenizer.mergeCount}</div>
          )}
          <div style={{ marginTop: 6, fontSize: 10, color: "#64748b" }}>
            {tokenizerMode === "bpe"
              ? "Subword tokens — can represent any text"
              : "Whole-word tokens — limited to known vocabulary"}
          </div>
        </div>

        <SectionLabel text="Training" />

        <SettingRow label={`Learning Rate: ${learningRate.toFixed(4)}`}>
          <input
            type="range"
            min={0.0001}
            max={useAdam ? 0.01 : 0.1}
            step={0.0001}
            value={learningRate}
            onChange={(e) => onLearningRateChange(Number(e.target.value))}
            style={{ width: "100%", accentColor: "#6366f1" }}
          />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "#475569", marginTop: 2 }}>
            <span>0.0001</span><span>{useAdam ? "0.01" : "0.1"}</span>
          </div>
        </SettingRow>

        <SettingRow label="Optimizer">
          <SegmentedControl
            options={["AdamW", "SGD"]}
            value={useAdam ? "AdamW" : "SGD"}
            onChange={(v) => {
              onUseAdamChange(v === "AdamW");
              if (v === "AdamW" && learningRate > 0.01) onLearningRateChange(0.001);
              if (v === "SGD" && learningRate < 0.005) onLearningRateChange(0.01);
            }}
          />
        </SettingRow>

        {useAdam && (
          <SettingRow label={`Weight Decay: ${weightDecay.toFixed(3)}`}>
            <input
              type="range"
              min={0}
              max={0.1}
              step={0.001}
              value={weightDecay}
              onChange={(e) => onWeightDecayChange(Number(e.target.value))}
              style={{ width: "100%", accentColor: "#6366f1" }}
            />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "#475569", marginTop: 2 }}>
              <span>0</span><span>0.1</span>
            </div>
          </SettingRow>
        )}

        <SettingRow label={`Batch Size: ${batchSize}`}>
          <SegmentedControl
            options={[4, 8, 16, 32, 64]}
            value={batchSize}
            onChange={onBatchSizeChange}
          />
        </SettingRow>

        <SectionLabel text="LR Schedule" />

        <SettingRow label="Warmup + Cosine Decay">
          <div
            onClick={() => onUseLRScheduleChange(!useLRSchedule)}
            style={{
              display: "flex", alignItems: "center", gap: 8, cursor: "pointer",
              padding: "6px 10px",
              background: useLRSchedule ? "rgba(99, 102, 241, 0.15)" : "rgba(255,255,255,0.04)",
              border: `1px solid ${useLRSchedule ? "rgba(99,102,241,0.3)" : "rgba(255,255,255,0.08)"}`,
              borderRadius: 6,
            }}
          >
            <div style={{
              width: 14, height: 14, borderRadius: 3,
              background: useLRSchedule ? "#6366f1" : "rgba(255,255,255,0.1)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 10, color: "white", fontWeight: 700,
            }}>
              {useLRSchedule ? "✓" : ""}
            </div>
            <span style={{ fontSize: 11, color: useLRSchedule ? "#a5b4fc" : "#64748b" }}>
              Enabled
            </span>
          </div>
        </SettingRow>

        {useLRSchedule && (
          <>
            <SettingRow label={`Warmup Steps: ${warmupSteps}`}>
              <input
                type="range"
                min={0}
                max={500}
                step={10}
                value={warmupSteps}
                onChange={(e) => onWarmupStepsChange(Number(e.target.value))}
                style={{ width: "100%", accentColor: "#6366f1" }}
              />
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "#475569", marginTop: 2 }}>
                <span>0</span><span>500</span>
              </div>
            </SettingRow>

            <SettingRow label={`Total Steps: ${totalSteps.toLocaleString()}`}>
              <input
                type="range"
                min={500}
                max={20000}
                step={500}
                value={totalSteps}
                onChange={(e) => onTotalStepsChange(Number(e.target.value))}
                style={{ width: "100%", accentColor: "#6366f1" }}
              />
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "#475569", marginTop: 2 }}>
                <span>500</span><span>20,000</span>
              </div>
            </SettingRow>
          </>
        )}

        <SectionLabel text="Training Corpus" />

        <SettingRow label="Corpus">
          <select
            value={corpusPreset}
            onChange={(e) => onCorpusChange(e.target.value)}
            style={{
              width: "100%",
              background: "rgba(255,255,255,0.06)",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 6,
              color: "#e2e8f0",
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 12,
              padding: "8px 10px",
              cursor: "pointer",
            }}
          >
            <option value="simple" style={{ background: "#1e293b" }}>Simple (6 sentences)</option>
            <option value="stories" style={{ background: "#1e293b" }}>Stories (100 sentences)</option>
            <option value="custom" style={{ background: "#1e293b" }}>Custom</option>
          </select>
        </SettingRow>

        <div style={{
          padding: 10, borderRadius: 8,
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,255,255,0.06)",
          fontSize: 11, color: "#94a3b8",
          lineHeight: 1.8,
          marginBottom: 12,
        }}>
          <div><span style={{ color: "#64748b" }}>Sentences:</span> {corpusStats.sentenceCount}</div>
          <div><span style={{ color: "#64748b" }}>Total words:</span> {corpusStats.totalWords.toLocaleString()}</div>
          <div><span style={{ color: "#64748b" }}>Unique words:</span> {corpusStats.uniqueWords}</div>
          <div><span style={{ color: "#64748b" }}>Avg length:</span> {corpusStats.avgLength} words/sentence</div>
        </div>

        {corpusPreset === "custom" && (
          <>
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
          </>
        )}

        <SectionLabel text="Checkpoints" />

        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={onSaveCheckpoint}
            style={{
              flex: 1,
              background: "linear-gradient(135deg, #3b82f6, #6366f1)",
              color: "white",
              border: "none",
              borderRadius: 8,
              padding: "10px 0",
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 11,
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            Save
          </button>
          <button
            onClick={onLoadCheckpoint}
            style={{
              flex: 1,
              background: "rgba(255,255,255,0.06)",
              color: "#94a3b8",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 8,
              padding: "10px 0",
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 11,
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            Load
          </button>
        </div>

        <div style={{ marginTop: 20, padding: 12, background: "rgba(255,255,255,0.03)", borderRadius: 8, border: "1px solid rgba(255,255,255,0.06)" }}>
          <div style={{ fontSize: 10, color: "#64748b", marginBottom: 4 }}>Model Parameters</div>
          <div style={{ fontSize: 18, color: "#a78bfa", fontWeight: 700 }}>{paramCount.toLocaleString()}</div>
          {estimatedParams !== paramCount && (
            <div style={{ fontSize: 11, color: "#64748b", marginTop: 4 }}>
              New config: ~{estimatedParams.toLocaleString()} params
            </div>
          )}
        </div>

        <div style={{ marginTop: 12, fontSize: 10, color: "#f97316", opacity: 0.7 }}>
          Changing architecture, tokenizer, or sentences resets training progress.
        </div>
      </div>
    </>
  );
}

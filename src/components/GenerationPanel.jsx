import { useState, useRef, useCallback, useEffect, memo } from "react";
import { generateStream } from "../model/generation.js";

function BlinkingCursor() {
  const [visible, setVisible] = useState(true);
  useEffect(() => {
    const id = setInterval(() => setVisible((v) => !v), 530);
    return () => clearInterval(id);
  }, []);
  return (
    <span style={{
      display: "inline-block",
      width: 8,
      height: 16,
      background: "#a78bfa",
      borderRadius: 2,
      opacity: visible ? 1 : 0,
      marginLeft: 2,
      alignSelf: "center",
    }} />
  );
}

function TokenBadge({ token, isPrompt }) {
  const probColor = isPrompt
    ? "#94a3b8"
    : token.probability < 0.2
      ? "#ef4444"
      : token.probability < 0.5
        ? "#fb923c"
        : "#4ade80";

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      <span style={{
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 13,
        color: isPrompt ? "#e2e8f0" : "#a78bfa",
        fontWeight: isPrompt ? 400 : 600,
        padding: "2px 4px",
        borderRadius: 4,
        background: isPrompt ? "transparent" : "rgba(167, 139, 250, 0.1)",
      }}>
        {token.text}
      </span>
      {!isPrompt && (
        <span style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 8,
          color: probColor,
          marginTop: 1,
        }}>
          {(token.probability * 100).toFixed(0)}%
        </span>
      )}
    </div>
  );
}

function SliderControl({ label, value, min, max, step, onChange, format, color }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.02)",
      borderRadius: 8,
      padding: "8px 12px",
      border: "1px solid rgba(255,255,255,0.05)",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 9,
          color: "#64748b",
          textTransform: "uppercase",
          letterSpacing: 1,
        }}>
          {label}
        </span>
        <span style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 10,
          color,
          fontWeight: 600,
        }}>
          {format(value)}
        </span>
      </div>
      <input
        type="range"
        min={min} max={max} step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: "100%", accentColor: color }}
      />
    </div>
  );
}

export default memo(function GenerationPanel({ paramsRef, config, tokenizer, isTraining }) {
  const [inputText, setInputText] = useState("");
  const [temperature, setTemperature] = useState(1.0);
  const [topK, setTopK] = useState(5);
  const [topP, setTopP] = useState(0.9);
  const [promptTokens, setPromptTokens] = useState([]);
  const [generatedTokens, setGeneratedTokens] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [inputError, setInputError] = useState(null);
  const abortRef = useRef(null);

  const isBPE = tokenizer.mergeCount > 0;

  const validateInput = useCallback((text) => {
    if (!text.trim()) return null;
    if (isBPE) return null;
    const words = text.trim().split(/\s+/);
    const unknowns = words.filter((w) => !(w in tokenizer.token2id));
    if (unknowns.length > 0) {
      return `Unknown word${unknowns.length > 1 ? "s" : ""}: ${unknowns.join(", ")}`;
    }
    return null;
  }, [tokenizer, isBPE]);

  const handleGenerate = useCallback(async () => {
    const params = paramsRef.current;
    if (!params || !inputText.trim()) return;
    const error = validateInput(inputText);
    if (error) return;

    // Cancel any ongoing generation
    if (abortRef.current) abortRef.current.abort();
    const abortController = new AbortController();
    abortRef.current = abortController;

    const encoded = tokenizer.encode(inputText.trim());
    const maxNewTokens = config.seqLen - encoded.length;
    if (maxNewTokens <= 0) {
      setInputError("Prompt is already at max sequence length");
      return;
    }

    setIsGenerating(true);
    setPromptTokens([]);
    setGeneratedTokens([]);

    try {
      for await (const event of generateStream(
        params, config, tokenizer, inputText.trim(),
        maxNewTokens, temperature, topK, topP,
        abortController.signal
      )) {
        if (abortController.signal.aborted) break;
        if (event.type === "prompt") {
          setPromptTokens(event.tokens);
        } else if (event.type === "generated") {
          setGeneratedTokens((prev) => [...prev, event.token]);
        }
      }
    } catch (e) {
      if (e.name !== "AbortError") console.error(e);
    }

    setIsGenerating(false);
  }, [paramsRef, config, tokenizer, inputText, temperature, topK, topP, validateInput]);

  const handleClear = useCallback(() => {
    if (abortRef.current) abortRef.current.abort();
    setPromptTokens([]);
    setGeneratedTokens([]);
    setInputText("");
    setInputError(null);
    setIsGenerating(false);
  }, []);

  const handleInputChange = useCallback((e) => {
    const val = e.target.value.toLowerCase();
    setInputText(val);
    setInputError(validateInput(val));
  }, [validateInput]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === "Enter") handleGenerate();
  }, [handleGenerate]);

  const btnBase = {
    borderRadius: 8,
    padding: "8px 16px",
    fontFamily: "'JetBrains Mono', monospace",
    fontSize: 11,
    fontWeight: 600,
    cursor: "pointer",
    border: "none",
  };

  const generateDisabled = isGenerating || !!inputError || !inputText.trim();

  return (
    <div style={{
      gridColumn: "1 / -1",
      background: "rgba(255,255,255,0.03)",
      borderRadius: 16,
      border: "1px solid rgba(255,255,255,0.08)",
      padding: "16px 20px",
    }}>
      {/* Header */}
      <div style={{
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        fontSize: 11,
        color: "#64748b",
        textTransform: "uppercase",
        letterSpacing: 2,
        marginBottom: 12,
      }}>
        Text Generation
      </div>

      {/* Input row */}
      <div style={{ display: "flex", gap: 8, marginBottom: inputError ? 4 : 12 }}>
        <input
          type="text"
          value={inputText}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder={isBPE ? "Type any text..." : "Type words from vocabulary..."}
          disabled={isGenerating}
          style={{
            flex: 1,
            background: "rgba(255,255,255,0.06)",
            border: inputError
              ? "1px solid rgba(239,68,68,0.5)"
              : "1px solid rgba(255,255,255,0.08)",
            borderRadius: 8,
            padding: "8px 12px",
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 12,
            color: "#e2e8f0",
            outline: "none",
          }}
        />
        <button
          onClick={handleGenerate}
          disabled={generateDisabled}
          style={{
            ...btnBase,
            background: generateDisabled
              ? "rgba(255,255,255,0.04)"
              : "linear-gradient(135deg, #3b82f6, #6366f1)",
            color: generateDisabled ? "#475569" : "white",
            cursor: generateDisabled ? "default" : "pointer",
          }}
        >
          {isGenerating ? "Generating..." : "Generate"}
        </button>
        <button
          onClick={handleClear}
          style={{
            ...btnBase,
            background: "rgba(255,255,255,0.06)",
            color: "#94a3b8",
            border: "1px solid rgba(255,255,255,0.08)",
          }}
        >
          Clear
        </button>
      </div>

      {/* Validation error */}
      {inputError && (
        <div style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 10,
          color: "#ef4444",
          marginBottom: 8,
        }}>
          {inputError}
        </div>
      )}

      {/* Token output area */}
      <div style={{
        minHeight: 60,
        background: "rgba(255,255,255,0.02)",
        borderRadius: 8,
        padding: 12,
        marginBottom: 12,
        display: "flex",
        flexWrap: "wrap",
        gap: 4,
        alignItems: "flex-end",
      }}>
        {promptTokens.length === 0 && generatedTokens.length === 0 && !isGenerating && (
          <span style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 11,
            color: "#475569",
          }}>
            Generated text will appear here...
          </span>
        )}
        {promptTokens.map((t, i) => (
          <TokenBadge key={`p-${i}`} token={t} isPrompt={true} />
        ))}
        {generatedTokens.map((t, i) => (
          <TokenBadge key={`g-${i}`} token={t} isPrompt={false} />
        ))}
        {isGenerating && <BlinkingCursor />}
      </div>

      {/* Sampling controls */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
        <SliderControl
          label="Temperature"
          value={temperature}
          min={0.1} max={2.0} step={0.1}
          onChange={setTemperature}
          format={(v) => v.toFixed(1)}
          color="#f59e0b"
        />
        <SliderControl
          label="Top-k"
          value={topK}
          min={1} max={Math.min(tokenizer.vocabSize, 50)} step={1}
          onChange={setTopK}
          format={(v) => String(v)}
          color="#60a5fa"
        />
        <SliderControl
          label="Top-p"
          value={topP}
          min={0.1} max={1.0} step={0.05}
          onChange={setTopP}
          format={(v) => v.toFixed(2)}
          color="#a78bfa"
        />
      </div>

      {/* Training hint */}
      {isTraining && (
        <div style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 9,
          color: "#64748b",
          marginTop: 8,
          textAlign: "center",
        }}>
          Model is training — generation uses current weights
        </div>
      )}
    </div>
  );
});

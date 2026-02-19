import { memo, useState } from "react";
import { CATEGORY_COLORS } from "../model/data.js";

const TOP_K = 10;

export default memo(function PredictionPanel({ probs, targetWord, inputWords, vocabSize, id2word, wordCategories, maxSpeed }) {
  const [showAll, setShowAll] = useState(false);

  const sortedIndices = [...Array(vocabSize).keys()].sort(
    (a, b) => probs[b] - probs[a]
  );

  const targetIdx = id2word.findIndex((w) => w === targetWord);
  const topKIndices = sortedIndices.slice(0, TOP_K);
  const targetInTopK = targetIdx >= 0 && topKIndices.includes(targetIdx);
  const targetRank = targetIdx >= 0 ? sortedIndices.indexOf(targetIdx) : -1;

  const indicesToRender = showAll ? sortedIndices : topKIndices;
  const isLargeVocab = vocabSize > 30;
  const labelWidth = isLargeVocab ? 72 : 48;

  const renderBar = (idx, showRank) => {
    const word = id2word[idx];
    const prob = probs[idx];
    const isTarget = word === targetWord;
    const isSubword = wordCategories[word] === "subword";
    const isFaint = prob < 0.01;
    const barColor = isTarget ? "#4ade80" : CATEGORY_COLORS[wordCategories[word]] || "#94a3b8";

    return (
      <div
        key={word}
        style={{
          display: "flex",
          alignItems: "center",
          marginBottom: 5,
          gap: 8,
        }}
      >
        {showRank !== undefined && (
          <div style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 9,
            color: "#334155",
            width: 24,
            textAlign: "right",
          }}>
            #{showRank + 1}
          </div>
        )}
        <div style={{
          width: labelWidth,
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 12,
          color: isFaint ? "#334155" : isTarget ? "#4ade80" : "#cbd5e1",
          textAlign: "right",
          fontWeight: isTarget ? 700 : 400,
          display: "flex",
          alignItems: "center",
          justifyContent: "flex-end",
          gap: 0,
          overflow: "hidden",
        }}>
          {isSubword && (
            <span style={{ color: "#c084fc", marginRight: 2, fontSize: 9, flexShrink: 0 }}>·</span>
          )}
          <span style={{
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}>
            {word}
          </span>
        </div>
        <div style={{
          flex: 1,
          height: 20,
          background: "rgba(255,255,255,0.04)",
          borderRadius: 4,
          overflow: "hidden",
          position: "relative",
        }}>
          <div
            style={{
              height: "100%",
              width: `${Math.max(prob * 100, isFaint ? 0.2 : 0.5)}%`,
              background: `linear-gradient(90deg, ${barColor}88, ${barColor})`,
              borderRadius: 4,
              transition: maxSpeed ? "none" : "width 0.3s ease",
              boxShadow: prob > 0.3 ? `0 0 12px ${barColor}44` : "none",
              opacity: isFaint ? 0.4 : 1,
            }}
          />
        </div>
        <div style={{
          width: 44,
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 11,
          color: isFaint ? "#334155" : isTarget ? "#4ade80" : "#64748b",
          fontWeight: isTarget ? 700 : 400,
        }}>
          {(prob * 100).toFixed(1)}%
        </div>
      </div>
    );
  };

  return (
    <div style={{
      background: "rgba(255,255,255,0.03)",
      borderRadius: 16,
      border: "1px solid rgba(255,255,255,0.08)",
      padding: "20px 24px",
      height: "100%",
      display: "flex",
      flexDirection: "column",
    }}>
      <div style={{ marginBottom: 8 }}>
        <div style={{
          fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
          fontSize: 11,
          color: "#64748b",
          textTransform: "uppercase",
          letterSpacing: 2,
          marginBottom: 8,
        }}>
          Prediction Probabilities
        </div>
        <div style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 12,
          color: "#94a3b8",
          marginBottom: 4,
        }}>
          Input: <span style={{ color: "#e2e8f0" }}>"{inputWords.join(" ")}"</span>
          {targetWord && (
            <span>
              {" → "}
              <span style={{ color: "#4ade80" }}>{targetWord}</span>
            </span>
          )}
        </div>
      </div>
      <div style={{ flex: 1, overflow: "auto" }}>
        {indicesToRender.map((idx) => renderBar(idx))}

        {/* Target not in top-K: show it separately */}
        {!showAll && !targetInTopK && targetIdx >= 0 && (
          <div style={{
            borderTop: "1px dashed rgba(74,222,128,0.3)",
            paddingTop: 6,
            marginTop: 6,
          }}>
            {renderBar(targetIdx, targetRank)}
          </div>
        )}

        {/* Show all / Show top-K toggle */}
        {vocabSize > TOP_K && (
          <button
            onClick={() => setShowAll((v) => !v)}
            style={{
              marginTop: 8,
              background: "none",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 6,
              padding: "4px 12px",
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 10,
              color: "#64748b",
              cursor: "pointer",
              width: "100%",
            }}
          >
            {showAll ? `Show top ${TOP_K}` : `Show all ${vocabSize} tokens`}
          </button>
        )}
      </div>
    </div>
  );
});

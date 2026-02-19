import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import { softmax } from "./model/math.js";
import { createTinyTransformer, forward, countParameters } from "./model/transformer.js";
import { trainStepMiniBatch, createAdamState, getLearningRate, verifyGradients } from "./model/training.js";
import { DEFAULT_SENTENCES, CORPUS_PRESETS, computeCorpusStats, buildWordTokenizer, buildBPETokenizer, CATEGORY_COLORS, makeTrainingData, generatePrompts } from "./model/data.js";
import { saveCheckpoint, loadCheckpoint } from "./model/checkpoint.js";
import { useTrainingWorker } from "./model/useTrainingWorker.js";

import PredictionPanel from "./components/PredictionPanel.jsx";
import EmbeddingPanel from "./components/EmbeddingPanel.jsx";
import AttentionPanel from "./components/AttentionPanel.jsx";
import LossPanel from "./components/LossPanel.jsx";
import MetricsPanel from "./components/MetricsPanel.jsx";
import SettingsPanel from "./components/SettingsPanel.jsx";
import TrainingLog from "./components/TrainingLog.jsx";

export default function App() {
  // Model config
  const [config, setConfig] = useState({
    embedDim: 16,
    numHeads: 2,
    ffnDim: 32,
    seqLen: 10,
    numBlocks: 1,
  });
  const [activation, setActivation] = useState("silu"); // "silu" | "gelu" | "relu"
  const [dropout, setDropout] = useState(0.0);
  const [learningRate, setLearningRate] = useState(0.001);
  const [batchSize, setBatchSize] = useState(32);
  const [useAdam, setUseAdam] = useState(true);
  const [useLRSchedule, setUseLRSchedule] = useState(true);
  const [warmupSteps, setWarmupSteps] = useState(100);
  const [totalSteps, setTotalSteps] = useState(5000);
  const [weightDecay, setWeightDecay] = useState(0.01);

  // Web Worker
  const { initWorker, trainBatch, supported: workerSupported } = useTrainingWorker();
  const [useWorker, setUseWorker] = useState(true);
  const useWorkerRef = useRef(true);
  const workerReadyRef = useRef(false);

  // Tokenizer
  const [tokenizerMode, setTokenizerMode] = useState("word"); // "word" | "bpe"
  const [bpeVocabSize, setBpeVocabSize] = useState(200);

  // Corpus & Sentences
  const [corpusPreset, setCorpusPreset] = useState("simple"); // "simple" | "stories" | "custom"
  const [sentences, setSentences] = useState(DEFAULT_SENTENCES);
  const [sentencesInput, setSentencesInput] = useState(DEFAULT_SENTENCES.join("\n"));

  // UI state
  const [showSettings, setShowSettings] = useState(false);
  const [step, setStep] = useState(0);
  const [lossHistory, setLossHistory] = useState([]);
  const [gradNormHistory, setGradNormHistory] = useState([]);
  const [lrHistory, setLrHistory] = useState([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [maxSpeed, setMaxSpeed] = useState(false);
  const [selectedPrompt, setSelectedPrompt] = useState(0);

  // Visualization state
  const [probs, setProbs] = useState([]);
  const [attnWeights, setAttnWeights] = useState([]);
  const [embeddingSnapshot, setEmbeddingSnapshot] = useState([]);

  // Training log
  const [logs, setLogs] = useState([]);

  // Refs
  const paramsRef = useRef(null);
  const playingRef = useRef(false);
  const trainingData = useRef([]);
  const stepRef = useRef(0);
  const lossBufferRef = useRef([]);       // Buffer losses between render frames
  const gradNormBufferRef = useRef([]);   // Buffer gradient norms between render frames
  const lrBufferRef = useRef([]);         // Buffer learning rates between render frames
  const vizDirtyRef = useRef(false);      // Flag: training happened since last render
  const selectedPromptRef = useRef(0);
  const maxSpeedRef = useRef(false);
  const learningRateRef = useRef(0.01);
  const activeConfigRef = useRef(null);
  const promptsRef = useRef([]);
  const id2wordRef = useRef({});
  const logBufferRef = useRef([]);
  const adamStateRef = useRef(null);
  const batchSizeRef = useRef(32);
  const useAdamRef = useRef(true);
  const useLRScheduleRef = useRef(true);
  const warmupStepsRef = useRef(100);
  const totalStepsRef = useRef(5000);
  const weightDecayRef = useRef(0.01);

  // Derived data
  const tokenizer = useMemo(
    () => tokenizerMode === "bpe"
      ? buildBPETokenizer(sentences, bpeVocabSize)
      : buildWordTokenizer(sentences),
    [sentences, tokenizerMode, bpeVocabSize]
  );
  const { vocabSize, id2token, token2id, tokenCategories, mergeCount } = tokenizer;
  // Backward-compat aliases used throughout the component
  const id2word = id2token;
  const wordCategories = tokenCategories;

  const prompts = useMemo(
    () => generatePrompts(sentences, tokenizer),
    [sentences, tokenizer]
  );

  const activeConfig = useMemo(() => ({
    ...config,
    vocabSize,
    activation,
    dropout,
  }), [config, vocabSize, activation, dropout]);

  const [paramCount, setParamCount] = useState(0);

  // Logging helper
  const addLog = useCallback((message, type = "info") => {
    setLogs((prev) => {
      const next = [...prev, { message, type }];
      if (next.length > 200) return next.slice(-200);
      return next;
    });
  }, []);

  // Initialize / reinitialize model when config or sentences change
  const configKey = JSON.stringify(activeConfig) + JSON.stringify(sentences) + tokenizerMode + bpeVocabSize;
  useEffect(() => {
    setIsPlaying(false);
    playingRef.current = false;
    paramsRef.current = createTinyTransformer(activeConfig);
    adamStateRef.current = createAdamState(paramsRef.current, activeConfig);
    trainingData.current = makeTrainingData(sentences, tokenizer);
    const count = countParameters(paramsRef.current);
    setParamCount(count);
    setStep(0);
    stepRef.current = 0;
    setLossHistory([]);
    setGradNormHistory([]);
    setLrHistory([]);
    setAttnWeights([]);
    setEmbeddingSnapshot([]);
    setProbs(Array(vocabSize).fill(1 / vocabSize));
    setSelectedPrompt((prev) => Math.min(prev, Math.max(0, prompts.length - 1)));

    const tokLabel = tokenizerMode === "bpe" ? `BPE(${vocabSize})` : `word(${vocabSize})`;
    const actLabel = (activeConfig.activation || "silu").toUpperCase();
    const dropLabel = activeConfig.dropout ? ` | drop=${activeConfig.dropout}` : "";
    addLog(`[Init] Model created: ${count.toLocaleString()} params | ${tokLabel} | ${activeConfig.numBlocks} block${activeConfig.numBlocks > 1 ? "s" : ""} | ${activeConfig.embedDim}d | ${activeConfig.numHeads} heads | ${actLabel}${dropLabel} | AdamW`, "config");

    // Verify gradients in development
    if (import.meta.env.DEV && trainingData.current.length > 0) {
      const { passed, maxRelError } = verifyGradients(paramsRef.current, trainingData.current, activeConfig);
      addLog(`[Verify] Gradient check: ${passed ? "PASSED" : "FAILED"} (max rel error: ${maxRelError.toExponential(2)})`, passed ? "success" : "warning");
    }

    // Run initial visualization
    if (prompts.length > 0) {
      const prompt = prompts[0];
      const result = forward(paramsRef.current, prompt.tokens, activeConfig);
      const probabilities = softmax(result.logits);
      setProbs(probabilities);
      setAttnWeights(result.attnWeights);
      setEmbeddingSnapshot(paramsRef.current.embedding.map((e) => [...e]));
    }

    // Initialize worker (if available)
    workerReadyRef.current = false;
    if (workerSupported) {
      const tokConfig = { mode: tokenizerMode, bpeVocabSize };
      initWorker(activeConfig, sentences, tokConfig).then((result) => {
        if (result) {
          workerReadyRef.current = true;
          addLog(`[Worker] Initialized`, "config");
        } else {
          workerReadyRef.current = false;
          setUseWorker(false);
        }
      });
    }
  }, [configKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // Keep refs in sync with state (avoids stale closures in training loop)
  selectedPromptRef.current = selectedPrompt;
  maxSpeedRef.current = maxSpeed;
  learningRateRef.current = learningRate;
  activeConfigRef.current = activeConfig;
  promptsRef.current = prompts;
  id2wordRef.current = id2word;
  batchSizeRef.current = batchSize;
  useAdamRef.current = useAdam;
  useLRScheduleRef.current = useLRSchedule;
  warmupStepsRef.current = warmupSteps;
  totalStepsRef.current = totalSteps;
  weightDecayRef.current = weightDecay;
  useWorkerRef.current = useWorker;

  // Update visualization (used for manual step & prompt changes)
  const updateVisualization = useCallback(
    (params, promptIdx) => {
      if (!prompts.length || !params) return;
      const prompt = prompts[promptIdx ?? selectedPrompt] || prompts[0];
      if (!prompt) return;
      const result = forward(params, prompt.tokens, activeConfig);
      const probabilities = softmax(result.logits);
      setProbs(probabilities);
      setAttnWeights(result.attnWeights);
      setEmbeddingSnapshot(params.embedding.map((e) => [...e]));
    },
    [selectedPrompt, prompts, activeConfig]
  );

  // Single manual training step (Step button)
  const doStep = useCallback(() => {
    if (!paramsRef.current) return;
    const cfg = activeConfigRef.current;
    const baseLR = learningRateRef.current;
    const lr = useLRScheduleRef.current
      ? getLearningRate(stepRef.current, warmupStepsRef.current, totalStepsRef.current, baseLR)
      : baseLR;
    const adam = useAdamRef.current ? adamStateRef.current : null;
    const wd = useAdamRef.current ? weightDecayRef.current : 0;
    const { loss, gradNorm } = trainStepMiniBatch(paramsRef.current, trainingData.current, cfg, lr, batchSizeRef.current, adam, wd);
    stepRef.current += 1;
    setStep(stepRef.current);
    setLossHistory((h) => {
      const next = [...h, loss];
      if (next.length > 500) return next.slice(-500);
      return next;
    });
    setGradNormHistory((h) => {
      const next = [...h, gradNorm];
      if (next.length > 500) return next.slice(-500);
      return next;
    });
    setLrHistory((h) => {
      const next = [...h, lr];
      if (next.length > 500) return next.slice(-500);
      return next;
    });
    updateVisualization(paramsRef.current, selectedPromptRef.current);

    const prompt = promptsRef.current[selectedPromptRef.current] || promptsRef.current[0];
    if (prompt) {
      const result = forward(paramsRef.current, prompt.tokens, cfg);
      const p = softmax(result.logits);
      let topIdx = 0;
      for (let i = 1; i < p.length; i++) { if (p[i] > p[topIdx]) topIdx = i; }
      const topWord = id2wordRef.current[topIdx];
      const topProb = p[topIdx];
      const correct = topWord === prompt.target;
      addLog(
        `[Step ${stepRef.current}] Loss: ${loss.toFixed(4)} | "${prompt.target}" → "${topWord}" (${(topProb * 100).toFixed(1)}%) ${correct ? "✓" : "✗"}`,
        correct ? "success" : "warning"
      );
    }
  }, [updateVisualization, addLog]);

  // Auto-play: decoupled training loop + rAF render loop
  useEffect(() => {
    playingRef.current = isPlaying;
    if (!isPlaying) return;

    let mounted = true;
    lossBufferRef.current = [];
    gradNormBufferRef.current = [];
    lrBufferRef.current = [];
    logBufferRef.current = [];
    vizDirtyRef.current = false;

    const shouldUseWorker = useWorkerRef.current && workerReadyRef.current;

    // --- Worker-based training loop ---
    const workerTrainLoop = () => {
      if (!mounted || !playingRef.current) return;

      const stepsPerBatch = maxSpeedRef.current ? 20 : 5;
      trainBatch({
        lr: learningRateRef.current,
        steps: stepsPerBatch,
        batchSize: batchSizeRef.current,
        weightDecay: useAdamRef.current ? weightDecayRef.current : 0,
        useAdam: useAdamRef.current,
        useLRSchedule: useLRScheduleRef.current,
        warmupSteps: warmupStepsRef.current,
        totalSteps: totalStepsRef.current,
        startStep: stepRef.current,
      }, (result) => {
        if (!mounted) return;

        // Update paramsRef with worker's snapshot (for viz)
        if (result.params) paramsRef.current = result.params;

        // Push to buffers
        for (let i = 0; i < result.losses.length; i++) {
          lossBufferRef.current.push(result.losses[i]);
          gradNormBufferRef.current.push(result.gradNorms[i]);
          lrBufferRef.current.push(result.lrs[i]);
        }

        stepRef.current += result.steps;
        vizDirtyRef.current = true;

        // Log entry from worker results
        if (result.params && result.losses.length > 0) {
          const lastLoss = result.losses[result.losses.length - 1];
          const prompt = promptsRef.current[selectedPromptRef.current] || promptsRef.current[0];
          if (prompt) {
            const cfg = activeConfigRef.current;
            const res = forward(result.params, prompt.tokens, cfg);
            const p = softmax(res.logits);
            let topIdx = 0;
            for (let k = 1; k < p.length; k++) { if (p[k] > p[topIdx]) topIdx = k; }
            const topWord = id2wordRef.current[topIdx];
            const topProb = p[topIdx];
            const correct = topWord === prompt.target;
            if (!maxSpeedRef.current || stepRef.current % 20 === 0) {
              logBufferRef.current.push({
                message: `[Step ${stepRef.current}] Loss: ${lastLoss.toFixed(4)} | "${prompt.target}" → "${topWord}" (${(topProb * 100).toFixed(1)}%) ${correct ? "✓" : "✗"}`,
                type: correct ? "success" : "warning",
              });
            }
          }

          // Convergence check every 50 steps
          if (stepRef.current % 50 < result.steps) {
            const allPrompts = promptsRef.current;
            const cfg = activeConfigRef.current;
            let allCorrect = allPrompts.length > 0;
            for (let pi = 0; pi < allPrompts.length; pi++) {
              const pr = allPrompts[pi];
              const res = forward(result.params, pr.tokens, cfg);
              const p = softmax(res.logits);
              let top = 0;
              for (let k = 1; k < p.length; k++) { if (p[k] > p[top]) top = k; }
              if (id2wordRef.current[top] !== pr.target) { allCorrect = false; break; }
            }
            if (allCorrect) {
              logBufferRef.current.push({
                message: `[Step ${stepRef.current}] All ${allPrompts.length} prompts correct — training complete`,
                type: "success",
              });
              playingRef.current = false;
              setIsPlaying(false);
              return; // Don't schedule next loop
            }
          }
        }

        // Continue loop
        if (mounted && playingRef.current) {
          if (maxSpeedRef.current) {
            setTimeout(workerTrainLoop, 0);
          } else {
            setTimeout(workerTrainLoop, Math.max(50, 500 / speed));
          }
        }
      });
    };

    // --- Main-thread training loop (fallback) ---
    const mainTrainLoop = () => {
      if (!mounted || !playingRef.current) return;
      const cfg = activeConfigRef.current;
      const params = paramsRef.current;
      if (!params || !cfg) return;

      const stepsPerFrame = maxSpeedRef.current ? 10 : 1;
      for (let i = 0; i < stepsPerFrame && mounted && playingRef.current; i++) {
        const baseLR = learningRateRef.current;
        const lr = useLRScheduleRef.current
          ? getLearningRate(stepRef.current, warmupStepsRef.current, totalStepsRef.current, baseLR)
          : baseLR;
        const adam = useAdamRef.current ? adamStateRef.current : null;
        const wd = useAdamRef.current ? weightDecayRef.current : 0;
        const { loss, gradNorm } = trainStepMiniBatch(params, trainingData.current, cfg, lr, batchSizeRef.current, adam, wd);
        stepRef.current += 1;
        lossBufferRef.current.push(loss);
        gradNormBufferRef.current.push(gradNorm);
        lrBufferRef.current.push(lr);
        vizDirtyRef.current = true;

        // Buffer log entries (every 20th step in maxSpeed, every step otherwise)
        if (!maxSpeedRef.current || stepRef.current % 20 === 0) {
          const prompt = promptsRef.current[selectedPromptRef.current] || promptsRef.current[0];
          if (prompt) {
            const result = forward(params, prompt.tokens, cfg);
            const p = softmax(result.logits);
            let topIdx = 0;
            for (let k = 1; k < p.length; k++) { if (p[k] > p[topIdx]) topIdx = k; }
            const topWord = id2wordRef.current[topIdx];
            const topProb = p[topIdx];
            const correct = topWord === prompt.target;
            logBufferRef.current.push({
              message: `[Step ${stepRef.current}] Loss: ${loss.toFixed(4)} | "${prompt.target}" → "${topWord}" (${(topProb * 100).toFixed(1)}%) ${correct ? "✓" : "✗"}`,
              type: correct ? "success" : "warning",
            });
          }
        }

        // Convergence check: every 50 steps, test all prompts
        if (stepRef.current % 50 === 0) {
          const allPrompts = promptsRef.current;
          let allCorrect = allPrompts.length > 0;
          for (let pi = 0; pi < allPrompts.length; pi++) {
            const pr = allPrompts[pi];
            const res = forward(params, pr.tokens, cfg);
            const p = softmax(res.logits);
            let top = 0;
            for (let k = 1; k < p.length; k++) { if (p[k] > p[top]) top = k; }
            if (id2wordRef.current[top] !== pr.target) { allCorrect = false; break; }
          }
          if (allCorrect) {
            logBufferRef.current.push({
              message: `[Step ${stepRef.current}] All ${allPrompts.length} prompts correct — training complete`,
              type: "success",
            });
            playingRef.current = false;
            setIsPlaying(false);
            break;
          }
        }
      }

      if (!playingRef.current) return;
      if (maxSpeedRef.current) {
        setTimeout(mainTrainLoop, 0);
      } else {
        setTimeout(mainTrainLoop, Math.max(50, 500 / speed));
      }
    };

    // Render loop — syncs buffered data to React state at screen refresh rate
    let rafId;
    const renderLoop = () => {
      if (!mounted) return;

      // Flush loss buffer to state
      if (lossBufferRef.current.length > 0) {
        const newLosses = lossBufferRef.current;
        lossBufferRef.current = [];
        setLossHistory((h) => {
          const next = h.concat(newLosses);
          if (next.length > 500) return next.slice(-500);
          return next;
        });
        setStep(stepRef.current);
      }

      // Flush gradient norm buffer
      if (gradNormBufferRef.current.length > 0) {
        const newNorms = gradNormBufferRef.current;
        gradNormBufferRef.current = [];
        setGradNormHistory((h) => {
          const next = h.concat(newNorms);
          if (next.length > 500) return next.slice(-500);
          return next;
        });
      }

      // Flush LR buffer
      if (lrBufferRef.current.length > 0) {
        const newLRs = lrBufferRef.current;
        lrBufferRef.current = [];
        setLrHistory((h) => {
          const next = h.concat(newLRs);
          if (next.length > 500) return next.slice(-500);
          return next;
        });
      }

      // Flush log buffer
      if (logBufferRef.current.length > 0) {
        const newLogs = logBufferRef.current;
        logBufferRef.current = [];
        setLogs((prev) => {
          const next = prev.concat(newLogs);
          if (next.length > 200) return next.slice(-200);
          return next;
        });
      }

      // Update visualization once per frame
      if (vizDirtyRef.current && paramsRef.current) {
        vizDirtyRef.current = false;
        const cfg = activeConfigRef.current;
        const prompt = promptsRef.current[selectedPromptRef.current] || promptsRef.current[0];
        if (prompt && cfg) {
          const result = forward(paramsRef.current, prompt.tokens, cfg);
          const probabilities = softmax(result.logits);
          setProbs(probabilities);
          setAttnWeights(result.attnWeights);
          setEmbeddingSnapshot(paramsRef.current.embedding.map((e) => [...e]));
        }
      }

      rafId = requestAnimationFrame(renderLoop);
    };

    // Choose training path
    if (shouldUseWorker) {
      workerTrainLoop();
    } else {
      mainTrainLoop();
    }
    rafId = requestAnimationFrame(renderLoop);

    return () => {
      mounted = false;
      cancelAnimationFrame(rafId);
    };
  }, [isPlaying, speed, trainBatch]);

  // Reset
  const handleReset = () => {
    setIsPlaying(false);
    playingRef.current = false;
    paramsRef.current = createTinyTransformer(activeConfig);
    adamStateRef.current = createAdamState(paramsRef.current, activeConfig);
    trainingData.current = makeTrainingData(sentences, tokenizer);
    setParamCount(countParameters(paramsRef.current));
    setStep(0);
    stepRef.current = 0;
    setLossHistory([]);
    setGradNormHistory([]);
    setLrHistory([]);
    setSelectedPrompt(0);
    if (prompts.length > 0) {
      updateVisualization(paramsRef.current, 0);
    }
    // Re-initialize worker so it also resets its params/adam state
    if (workerSupported) {
      workerReadyRef.current = false;
      const tokConfig = { mode: tokenizerMode, bpeVocabSize };
      initWorker(activeConfig, sentences, tokConfig).then((result) => {
        workerReadyRef.current = !!result;
      });
    }
    addLog("[Reset] Model re-initialized", "config");
  };

  // Prompt change
  const handlePromptChange = (idx) => {
    setSelectedPrompt(idx);
    if (paramsRef.current) {
      updateVisualization(paramsRef.current, idx);
    }
  };

  // Snap seqLen to valid discrete options
  const SEQ_LEN_OPTIONS = [10, 16, 24, 32, 48];
  const snapSeqLen = (minRequired) => SEQ_LEN_OPTIONS.find((v) => v >= minRequired) || 48;

  // Switch corpus preset
  const handleCorpusChange = useCallback((preset) => {
    setCorpusPreset(preset);
    if (preset === "custom") return; // keep current sentences until user applies
    const corpus = CORPUS_PRESETS[preset];
    if (!corpus) return;
    const newSentences = corpus.sentences;
    const stats = computeCorpusStats(newSentences);
    const maxLen = Math.max(...newSentences.map((s) => s.split(/\s+/).length));
    setSentences(newSentences);
    setSentencesInput(newSentences.join("\n"));
    setConfig((prev) => ({ ...prev, seqLen: snapSeqLen(maxLen + 2) }));
    addLog(`[Config] Loaded '${corpus.label}' corpus (${stats.sentenceCount} sentences, ${stats.uniqueWords} unique words)`, "config");
  }, [addLog]);

  // Apply custom sentences
  const handleApplySentences = () => {
    const lines = sentencesInput
      .split("\n")
      .map((s) => s.trim().toLowerCase())
      .filter((s) => s.split(/\s+/).length >= 2);
    if (lines.length === 0) return;
    const maxLen = Math.max(...lines.map((s) => s.split(/\s+/).length));
    setSentences(lines);
    setCorpusPreset("custom");
    setConfig((prev) => ({ ...prev, seqLen: snapSeqLen(Math.max(prev.seqLen, maxLen + 2)) }));
    setShowSettings(false);
    addLog(`[Config] Applied ${lines.length} custom sentences`, "config");
  };

  // Checkpoint save/load
  const handleSaveCheckpoint = useCallback(() => {
    const json = saveCheckpoint(
      paramsRef.current, adamStateRef.current, stepRef.current,
      activeConfigRef.current, lossHistory
    );
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `checkpoint-step-${stepRef.current}.json`;
    a.click();
    URL.revokeObjectURL(url);
    addLog(`[Checkpoint] Saved at step ${stepRef.current}`, "config");
  }, [lossHistory, addLog]);

  const handleLoadCheckpoint = useCallback(() => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.onchange = (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        try {
          const checkpoint = loadCheckpoint(ev.target.result);
          if (checkpoint.config.vocabSize !== activeConfigRef.current.vocabSize) {
            addLog("[Checkpoint] Error: vocab size mismatch", "warning");
            return;
          }
          setIsPlaying(false);
          playingRef.current = false;
          paramsRef.current = checkpoint.params;
          adamStateRef.current = checkpoint.adamState;
          stepRef.current = checkpoint.step;
          setStep(checkpoint.step);
          setLossHistory(checkpoint.lossHistory);
          setGradNormHistory([]);
          setLrHistory([]);
          updateVisualization(paramsRef.current, selectedPromptRef.current);
          addLog(`[Checkpoint] Loaded at step ${checkpoint.step}`, "config");
        } catch (err) {
          addLog(`[Checkpoint] Load failed: ${err.message}`, "warning");
        }
      };
      reader.readAsText(file);
    };
    input.click();
  }, [addLog, updateVisualization]);

  // Current prediction info
  const currentPrompt = prompts[selectedPrompt] || prompts[0] || { tokens: [], label: "", target: "" };
  const inputWords = currentPrompt.tokens.map((t) => id2word[t]);
  const topPrediction = probs.length > 0 ? id2word[probs.indexOf(Math.max(...probs))] : "";
  const topProb = probs.length > 0 ? Math.max(...probs) : 0;
  const isCorrect = topPrediction === currentPrompt.target;

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0a0e17",
      color: "#e2e8f0",
      fontFamily: "'Inter', 'SF Pro Display', -apple-system, sans-serif",
      padding: 24,
    }}>
      {showSettings && (
        <SettingsPanel
          config={config}
          onConfigChange={(newConfig) => {
            setIsPlaying(false);
            setConfig(newConfig);
          }}
          activation={activation}
          onActivationChange={(v) => { setIsPlaying(false); setActivation(v); }}
          dropout={dropout}
          onDropoutChange={(v) => { setIsPlaying(false); setDropout(v); }}
          learningRate={learningRate}
          onLearningRateChange={setLearningRate}
          batchSize={batchSize}
          onBatchSizeChange={setBatchSize}
          useAdam={useAdam}
          onUseAdamChange={setUseAdam}
          useLRSchedule={useLRSchedule}
          onUseLRScheduleChange={setUseLRSchedule}
          warmupSteps={warmupSteps}
          onWarmupStepsChange={setWarmupSteps}
          totalSteps={totalSteps}
          onTotalStepsChange={setTotalSteps}
          corpusPreset={corpusPreset}
          onCorpusChange={handleCorpusChange}
          sentences={sentences}
          sentencesInput={sentencesInput}
          onSentencesInputChange={setSentencesInput}
          onApplySentences={handleApplySentences}
          weightDecay={weightDecay}
          onWeightDecayChange={setWeightDecay}
          tokenizerMode={tokenizerMode}
          onTokenizerModeChange={setTokenizerMode}
          bpeVocabSize={bpeVocabSize}
          onBpeVocabSizeChange={setBpeVocabSize}
          tokenizer={tokenizer}
          onSaveCheckpoint={handleSaveCheckpoint}
          onLoadCheckpoint={handleLoadCheckpoint}
          paramCount={paramCount}
          onClose={() => setShowSettings(false)}
        />
      )}

      <div style={{ maxWidth: 1200, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 24, textAlign: "center" }}>
          <h1 style={{
            fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
            fontSize: 24,
            fontWeight: 700,
            background: "linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            marginBottom: 6,
          }}>
            Tiny Transformer — Live Training
          </h1>
          <p style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 12,
            color: "#475569",
          }}>
            {vocabSize} {tokenizerMode === "bpe" ? "BPE tokens" : "words"} · {activeConfig.embedDim}d · {activeConfig.numHeads} heads · {activeConfig.numBlocks} block{activeConfig.numBlocks > 1 ? "s" : ""} · {paramCount.toLocaleString()} params · {useAdam ? "AdamW" : "SGD"} lr={learningRate} · batch={batchSize}
          </p>
        </div>

        {/* Controls */}
        <div style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 12,
          marginBottom: 20,
          flexWrap: "wrap",
        }}>
          <button
            onClick={() => setShowSettings(true)}
            style={{
              background: "rgba(255,255,255,0.06)",
              color: "#94a3b8",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 10,
              padding: "10px 20px",
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 13,
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            ⚙ Settings
          </button>
          <button
            onClick={doStep}
            disabled={isPlaying}
            style={{
              background: isPlaying ? "rgba(255,255,255,0.04)" : "linear-gradient(135deg, #3b82f6, #6366f1)",
              color: isPlaying ? "#475569" : "white",
              border: "none",
              borderRadius: 10,
              padding: "10px 20px",
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 13,
              fontWeight: 600,
              cursor: isPlaying ? "default" : "pointer",
              letterSpacing: 0.5,
            }}
          >
            ▶ Step
          </button>
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            style={{
              background: isPlaying ? "linear-gradient(135deg, #ef4444, #f97316)" : "linear-gradient(135deg, #22c55e, #10b981)",
              color: "white",
              border: "none",
              borderRadius: 10,
              padding: "10px 20px",
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 13,
              fontWeight: 600,
              cursor: "pointer",
              letterSpacing: 0.5,
            }}
          >
            {isPlaying ? "⏸ Pause" : "⏵ Play"}
          </button>
          <button
            onClick={handleReset}
            style={{
              background: "rgba(255,255,255,0.06)",
              color: "#94a3b8",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 10,
              padding: "10px 20px",
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 13,
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            ↺ Reset
          </button>

          <div style={{
            display: "flex", alignItems: "center", gap: 8,
            background: "rgba(255,255,255,0.04)",
            borderRadius: 10,
            padding: "6px 16px",
            border: "1px solid rgba(255,255,255,0.06)",
          }}>
            <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#64748b" }}>Speed</span>
            <input
              type="range"
              min={1}
              max={10}
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              disabled={maxSpeed}
              style={{ width: 80, accentColor: "#6366f1", opacity: maxSpeed ? 0.3 : 1 }}
            />
            <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#94a3b8", width: 24 }}>
              {maxSpeed ? "∞" : `${speed}x`}
            </span>
          </div>

          <button
            onClick={() => setMaxSpeed(!maxSpeed)}
            style={{
              background: maxSpeed ? "linear-gradient(135deg, #f59e0b, #f97316)" : "rgba(255,255,255,0.06)",
              color: maxSpeed ? "#0f172a" : "#64748b",
              border: maxSpeed ? "none" : "1px solid rgba(255,255,255,0.08)",
              borderRadius: 10,
              padding: "10px 14px",
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 11,
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            {maxSpeed ? "⚡ MAX" : "⚡ Max"}
          </button>

          <div style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 14,
            color: "#a78bfa",
            fontWeight: 700,
            background: "rgba(167, 139, 250, 0.08)",
            padding: "8px 16px",
            borderRadius: 10,
            border: "1px solid rgba(167, 139, 250, 0.15)",
          }}>
            Step {step}
          </div>
        </div>

        {/* Prompt Selector */}
        <div style={{
          display: "flex",
          justifyContent: "center",
          gap: 8,
          marginBottom: 20,
          flexWrap: "wrap",
        }}>
          {prompts.map((p, i) => (
            <button
              key={i}
              onClick={() => handlePromptChange(i)}
              style={{
                background: selectedPrompt === i ? "rgba(96, 165, 250, 0.15)" : "rgba(255,255,255,0.04)",
                color: selectedPrompt === i ? "#60a5fa" : "#64748b",
                border: `1px solid ${selectedPrompt === i ? "rgba(96,165,250,0.3)" : "rgba(255,255,255,0.06)"}`,
                borderRadius: 8,
                padding: "6px 14px",
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 11,
                cursor: "pointer",
              }}
            >
              {p.label}
            </button>
          ))}
        </div>

        {/* Status banner */}
        {step > 0 && currentPrompt.target && (
          <div style={{
            textAlign: "center",
            marginBottom: 16,
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 13,
            padding: "8px 16px",
            borderRadius: 10,
            background: isCorrect ? "rgba(74, 222, 128, 0.08)" : "rgba(251, 146, 60, 0.08)",
            border: `1px solid ${isCorrect ? "rgba(74,222,128,0.2)" : "rgba(251,146,60,0.15)"}`,
            color: isCorrect ? "#4ade80" : "#fb923c",
          }}>
            Prediction: <strong>{topPrediction}</strong> ({(topProb * 100).toFixed(1)}%)
            {isCorrect ? " ✓ Correct!" : ` ✗ Expected: ${currentPrompt.target}`}
          </div>
        )}

        {/* Dashboard Grid */}
        <div style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 16,
          maxWidth: 1200,
          margin: "0 auto",
        }}>
          <PredictionPanel
            probs={probs}
            targetWord={currentPrompt.target}
            inputWords={inputWords}
            vocabSize={vocabSize}
            id2word={id2word}
            wordCategories={wordCategories}
            maxSpeed={maxSpeed}
          />
          <EmbeddingPanel
            embeddings={embeddingSnapshot.length > 0 ? embeddingSnapshot : Array.from({ length: vocabSize }, () => Array(activeConfig.embedDim).fill(0))}
            id2word={id2word}
            wordCategories={wordCategories}
          />
          <AttentionPanel
            attnWeights={attnWeights}
            inputWords={inputWords}
            numHeads={activeConfig.numHeads}
            numBlocks={activeConfig.numBlocks}
            maxSpeed={maxSpeed}
          />
          <LossPanel lossHistory={lossHistory} vocabSize={vocabSize} />
          <MetricsPanel
            lossHistory={lossHistory}
            gradNormHistory={gradNormHistory}
            lrHistory={lrHistory}
            vocabSize={vocabSize}
            currentStep={step}
            warmupSteps={warmupSteps}
            totalSteps={totalSteps}
            baseLR={learningRate}
            useLRSchedule={useLRSchedule}
          />
          <TrainingLog logs={logs} />
        </div>

        {/* Footer */}
        <div style={{
          textAlign: "center",
          marginTop: 20,
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 10,
          color: "#334155",
        }}>
          Pure JavaScript transformer · No ML frameworks · Backpropagation · {vocabSize} tokens × {activeConfig.embedDim}d × {activeConfig.numBlocks} block{activeConfig.numBlocks > 1 ? "s" : ""}
        </div>
      </div>
    </div>
  );
}

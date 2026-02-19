import { useRef, useState, useCallback, useEffect } from "react";

export function useTrainingWorker() {
  const workerRef = useRef(null);
  const [supported] = useState(() => typeof Worker !== "undefined");
  const [ready, setReady] = useState(false);
  const onTrainResultRef = useRef(null);
  const pendingRef = useRef(null); // For one-shot requests (init, forward)

  useEffect(() => {
    if (!supported) return;
    try {
      workerRef.current = new Worker(
        new URL("./training.worker.js", import.meta.url),
        { type: "module" }
      );
      workerRef.current.onmessage = (e) => {
        const { type, payload } = e.data;
        if (type === "trainBatchResult") {
          onTrainResultRef.current?.(payload);
        } else if (pendingRef.current && pendingRef.current.type === type) {
          pendingRef.current.resolve(payload);
          pendingRef.current = null;
        }
      };
      workerRef.current.onerror = () => {
        workerRef.current = null;
        setReady(false);
      };
      setReady(true);
    } catch {
      workerRef.current = null;
      setReady(false);
    }
    return () => {
      workerRef.current?.terminate();
      workerRef.current = null;
      setReady(false);
    };
  }, [supported]);

  const sendAndWait = useCallback((type, payload, responseType) => {
    return new Promise((resolve) => {
      if (!workerRef.current) {
        resolve(null);
        return;
      }
      pendingRef.current = { type: responseType, resolve };
      workerRef.current.postMessage({ type, payload });
    });
  }, []);

  const initWorker = useCallback((config, sentences, tokenizerConfig) => {
    return sendAndWait("init", { config, sentences, tokenizerConfig }, "initResult");
  }, [sendAndWait]);

  // trainBatch: callback-based for streaming results
  const trainBatch = useCallback((payload, onResult) => {
    onTrainResultRef.current = onResult;
    if (workerRef.current) {
      workerRef.current.postMessage({ type: "trainBatch", payload });
    }
  }, []);

  const forwardPass = useCallback((tokens) => {
    return sendAndWait("forward", { tokens }, "forwardResult");
  }, [sendAndWait]);

  return {
    initWorker,
    trainBatch,
    forwardPass,
    supported: supported && ready,
    workerRef,
  };
}

import { useRef, useState, useCallback, useEffect } from "react";

export function useTrainingWorker() {
  const workerRef = useRef(null);
  const [supported] = useState(() => typeof Worker !== "undefined");
  const pendingRef = useRef(null);

  useEffect(() => {
    if (!supported) return;
    try {
      workerRef.current = new Worker(
        new URL("./training.worker.js", import.meta.url),
        { type: "module" }
      );
      workerRef.current.onmessage = (e) => {
        const { type, payload } = e.data;
        if (pendingRef.current && pendingRef.current.type === type) {
          pendingRef.current.resolve(payload);
          pendingRef.current = null;
        }
      };
    } catch {
      workerRef.current = null;
    }
    return () => {
      workerRef.current?.terminate();
      workerRef.current = null;
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

  const initWorker = useCallback((config, sentences) => {
    return sendAndWait("init", { config, sentences }, "initResult");
  }, [sendAndWait]);

  const train = useCallback((lr, steps = 1) => {
    return sendAndWait("train", { lr, steps }, "trainResult");
  }, [sendAndWait]);

  const forwardPass = useCallback((tokens) => {
    return sendAndWait("forward", { tokens }, "forwardResult");
  }, [sendAndWait]);

  const setParams = useCallback((params, config, trainingData) => {
    if (workerRef.current) {
      workerRef.current.postMessage({ type: "setParams", payload: { params, config, trainingData } });
    }
  }, []);

  return {
    initWorker,
    train,
    forwardPass,
    setParams,
    supported: supported && workerRef.current !== null,
    workerRef,
  };
}

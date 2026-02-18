import { trainStepBackprop } from "./training.js";
import { forward } from "./transformer.js";
import { softmax } from "./math.js";
import { createTinyTransformer } from "./transformer.js";
import { makeTrainingData, buildVocabulary } from "./data.js";

let params = null;
let trainingData = null;
let config = null;

self.onmessage = (e) => {
  const { type, payload } = e.data;

  switch (type) {
    case "init": {
      // Rebuild model and data from config + sentences
      config = payload.config;
      params = createTinyTransformer(config);
      const vocab = buildVocabulary(payload.sentences);
      trainingData = makeTrainingData(payload.sentences, vocab.word2id);
      self.postMessage({
        type: "initResult",
        payload: {
          params,
          paramCount: countParams(params),
        },
      });
      break;
    }

    case "setParams": {
      // Receive updated params from main thread (e.g. after reset)
      params = payload.params;
      config = payload.config;
      trainingData = payload.trainingData;
      break;
    }

    case "train": {
      if (!params || !trainingData || !config) {
        self.postMessage({ type: "trainResult", payload: { losses: [], steps: 0 } });
        break;
      }
      const { lr, steps } = payload;
      const losses = [];
      for (let i = 0; i < steps; i++) {
        const loss = trainStepBackprop(params, trainingData, config, lr);
        losses.push(loss);
      }
      self.postMessage({
        type: "trainResult",
        payload: { params, losses },
      });
      break;
    }

    case "forward": {
      if (!params || !config) {
        self.postMessage({ type: "forwardResult", payload: null });
        break;
      }
      const { tokens } = payload;
      const result = forward(params, tokens, config);
      const probs = softmax(result.logits);
      self.postMessage({
        type: "forwardResult",
        payload: {
          probs,
          attnWeights: result.attnWeights,
          embeddings: params.embedding.map((e) => [...e]),
        },
      });
      break;
    }
  }
};

function countParams(p) {
  let count = 0;
  function c(arr) {
    if (arr[0]?.length !== undefined) arr.forEach((r) => (count += r.length));
    else count += arr.length;
  }
  c(p.embedding); c(p.Wout); c(p.bout);
  for (const block of p.blocks) {
    for (const key of Object.keys(block)) c(block[key]);
  }
  return count;
}

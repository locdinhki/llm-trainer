import { trainStepMiniBatch, createAdamState, getLearningRate } from "./training.js";
import { createTinyTransformer, forward, countParameters } from "./transformer.js";
import { softmax } from "./math.js";
import { makeTrainingData, buildWordTokenizer, buildBPETokenizer } from "./data.js";

let params = null;
let adamState = null;
let trainingData = null;
let config = null;

self.onmessage = (e) => {
  const { type, payload } = e.data;

  switch (type) {
    case "init": {
      const { config: cfg, sentences, tokenizerConfig } = payload;
      config = cfg;
      const tokenizer = tokenizerConfig.mode === "bpe"
        ? buildBPETokenizer(sentences, tokenizerConfig.bpeVocabSize)
        : buildWordTokenizer(sentences);
      config = { ...config, vocabSize: tokenizer.vocabSize };
      params = createTinyTransformer(config);
      adamState = createAdamState(params, config);
      trainingData = makeTrainingData(sentences, tokenizer);
      self.postMessage({
        type: "initResult",
        payload: { paramCount: countParameters(params) },
      });
      break;
    }

    case "trainBatch": {
      if (!params || !trainingData || !config) {
        self.postMessage({ type: "trainBatchResult", payload: { losses: [], gradNorms: [], lrs: [], steps: 0 } });
        break;
      }
      const { lr, steps, batchSize, weightDecay, useAdam, useLRSchedule, warmupSteps, totalSteps, startStep } = payload;
      const losses = [];
      const gradNorms = [];
      const lrs = [];
      let currentStep = startStep;

      for (let i = 0; i < steps; i++) {
        const effectiveLR = useLRSchedule
          ? getLearningRate(currentStep, warmupSteps, totalSteps, lr)
          : lr;
        const adam = useAdam ? adamState : null;
        const wd = useAdam ? weightDecay : 0;
        const { loss, gradNorm } = trainStepMiniBatch(params, trainingData, config, effectiveLR, batchSize, adam, wd);
        losses.push(loss);
        gradNorms.push(gradNorm);
        lrs.push(effectiveLR);
        currentStep += 1;
      }

      // Send results back (params snapshot for visualization)
      self.postMessage({
        type: "trainBatchResult",
        payload: { losses, gradNorms, lrs, steps, params },
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

    case "setConfig": {
      if (payload.config) {
        config = { ...config, ...payload.config };
      }
      break;
    }
  }
};

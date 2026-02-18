import { matVec, vecAdd, layerNorm, softmax, relu } from "./math.js";

// Forward pass that caches all intermediates needed for backward pass.
// Mirrors the logic in transformer.js forward() exactly.
export function forwardWithCache(params, tokens, config) {
  const { embedDim, numHeads, numBlocks = 1 } = config;
  const headDim = embedDim / numHeads;
  const len = tokens.length;

  // Embedding lookup
  let hidden = tokens.map((t, pos) =>
    vecAdd(params.embedding[t], params.posEmbedding[pos])
  );

  const blockCaches = [];

  for (let blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
    const block = params.blocks[blockIdx];
    const bc = {};

    // Cache pre-LN1 hidden states (input to this block)
    bc.preNorm1 = hidden.map((h) => h.slice());

    // Layer norm 1
    bc.normed1 = hidden.map((h) => layerNorm(h, block.ln1_g, block.ln1_b));

    // Q, K, V projections
    bc.Q = bc.normed1.map((h) => vecAdd(matVec(block.Wq, h), block.bq));
    bc.K = bc.normed1.map((h) => vecAdd(matVec(block.Wk, h), block.bk));
    bc.V = bc.normed1.map((h) => vecAdd(matVec(block.Wv, h), block.bv));

    // Multi-head causal attention
    bc.attnOutputs = Array.from({ length: len }, () => new Float32Array(embedDim));
    bc.attnWeights = []; // [head][queryPos][keyPos]

    for (let head = 0; head < numHeads; head++) {
      const offset = head * headDim;
      const headWeights = [];

      for (let i = 0; i < len; i++) {
        const scores = [];
        for (let j = 0; j <= i; j++) {
          let score = 0;
          for (let d = 0; d < headDim; d++) {
            score += bc.Q[i][offset + d] * bc.K[j][offset + d];
          }
          scores.push(score / Math.sqrt(headDim));
        }
        for (let j = i + 1; j < len; j++) {
          scores.push(-1e9);
        }

        const weights = softmax(scores);
        headWeights.push(weights);

        for (let d = 0; d < headDim; d++) {
          let val = 0;
          for (let j = 0; j < len; j++) {
            val += weights[j] * bc.V[j][offset + d];
          }
          bc.attnOutputs[i][offset + d] = val;
        }
      }
      bc.attnWeights.push(headWeights);
    }

    // Wo projection + first residual
    const attnProj = bc.attnOutputs.map((h) => vecAdd(matVec(block.Wo, h), block.bo));
    hidden = hidden.map((h, i) => vecAdd(h, attnProj[i]));

    // Cache pre-LN2 hidden states
    bc.preNorm2 = hidden.map((h) => h.slice());

    // Layer norm 2
    bc.normed2 = hidden.map((h) => layerNorm(h, block.ln2_g, block.ln2_b));

    // FFN: relu(W1*normed2 + b1), then W2*ffn1 + b2
    bc.ffn1Pre = bc.normed2.map((h) => vecAdd(matVec(block.W1, h), block.b1));
    bc.ffn1 = bc.ffn1Pre.map((h) => {
      const n = h.length;
      const result = new Float32Array(n);
      for (let i = 0; i < n; i++) result[i] = h[i] > 0 ? h[i] : 0;
      return result;
    });
    const ffn2 = bc.ffn1.map((h) => vecAdd(matVec(block.W2, h), block.b2));

    // Second residual
    hidden = hidden.map((h, i) => vecAdd(h, ffn2[i]));

    blockCaches.push(bc);
  }

  const lastHidden = hidden[hidden.length - 1];
  const logits = vecAdd(matVec(params.Wout, lastHidden), params.bout);

  return {
    logits,
    cache: {
      tokens,
      blocks: blockCaches,
      lastHidden,
      logits,
    },
  };
}

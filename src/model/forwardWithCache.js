import { matVec, vecAdd, vecMul, rmsNorm, softmax, silu, gelu, relu } from "./math.js";
import { precomputeFreqs, applyRoPE } from "./rope.js";

// Forward pass that caches all intermediates needed for backward pass.
// Mirrors the logic in transformer.js forward() exactly.
export function forwardWithCache(params, tokens, config) {
  const { embedDim, numHeads, numBlocks = 1 } = config;
  const headDim = embedDim / numHeads;
  const len = tokens.length;

  // Embedding lookup (no positional embedding — RoPE handles position)
  let hidden = tokens.map((t) => params.embedding[t].slice());

  const freqs = precomputeFreqs(len, headDim);
  const blockCaches = [];

  for (let blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
    const block = params.blocks[blockIdx];
    const bc = {};

    // Cache pre-RMSNorm1 hidden states (input to this block)
    bc.preNorm1 = hidden.map((h) => h.slice());

    // RMSNorm 1
    bc.normed1 = hidden.map((h) => rmsNorm(h, block.ln1_g));

    // Q, K, V projections (pre-RoPE)
    bc.Q = bc.normed1.map((h) => vecAdd(matVec(block.Wq, h), block.bq));
    bc.K = bc.normed1.map((h) => vecAdd(matVec(block.Wk, h), block.bk));
    bc.V = bc.normed1.map((h) => vecAdd(matVec(block.Wv, h), block.bv));

    // Apply RoPE to Q and K per-head (attention uses rotated Q/K)
    for (let head = 0; head < numHeads; head++) {
      const off = head * headDim;
      for (let pos = 0; pos < len; pos++) {
        const qSlice = bc.Q[pos].slice(off, off + headDim);
        const kSlice = bc.K[pos].slice(off, off + headDim);
        bc.Q[pos].set(applyRoPE(qSlice, pos, freqs), off);
        bc.K[pos].set(applyRoPE(kSlice, pos, freqs), off);
      }
    }

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

    // Wo projection + first residual (with optional dropout)
    const attnProj = bc.attnOutputs.map((h) => vecAdd(matVec(block.Wo, h), block.bo));
    const dropoutRate = config.dropout || 0;
    if (dropoutRate > 0) {
      const scale = 1 / (1 - dropoutRate);
      bc.attnDropoutMask = attnProj.map((h) => {
        const mask = new Float32Array(h.length);
        for (let i = 0; i < h.length; i++) mask[i] = Math.random() < dropoutRate ? 0 : scale;
        return mask;
      });
      hidden = hidden.map((h, i) => vecAdd(h, vecMul(attnProj[i], bc.attnDropoutMask[i])));
    } else {
      bc.attnDropoutMask = null;
      hidden = hidden.map((h, i) => vecAdd(h, attnProj[i]));
    }

    // Cache pre-LN2 hidden states
    bc.preNorm2 = hidden.map((h) => h.slice());

    // Layer norm 2
    bc.normed2 = hidden.map((h) => rmsNorm(h, block.ln2_g));

    // GLU FFN: gate = activation(W_gate @ x), up = W1 @ x, out = W2 @ (gate * up)
    const activationFn = { silu, gelu, relu }[config.activation] || silu;
    bc.gatePre = bc.normed2.map((h) => vecAdd(matVec(block.W_gate, h), block.b_gate));
    bc.gate = bc.gatePre.map((h) => activationFn(h));
    bc.up = bc.normed2.map((h) => vecAdd(matVec(block.W1, h), block.b1));
    bc.gated = bc.gate.map((g, i) => vecMul(g, bc.up[i]));
    const ffn2 = bc.gated.map((h) => vecAdd(matVec(block.W2, h), block.b2));

    // Second residual (with optional dropout)
    if (dropoutRate > 0) {
      const scale = 1 / (1 - dropoutRate);
      bc.ffnDropoutMask = ffn2.map((h) => {
        const mask = new Float32Array(h.length);
        for (let i = 0; i < h.length; i++) mask[i] = Math.random() < dropoutRate ? 0 : scale;
        return mask;
      });
      hidden = hidden.map((h, i) => vecAdd(h, vecMul(ffn2[i], bc.ffnDropoutMask[i])));
    } else {
      bc.ffnDropoutMask = null;
      hidden = hidden.map((h, i) => vecAdd(h, ffn2[i]));
    }

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

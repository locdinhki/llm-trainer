import { mulberry32, matVec, vecAdd, vecMul, rmsNorm, softmax, silu, gelu, relu } from "./math.js";
import { precomputeFreqs, applyRoPE } from "./rope.js";

export function createTinyTransformer(config) {
  const { vocabSize, embedDim, ffnDim, seqLen, numBlocks = 1 } = config;
  const rand = mulberry32(42);
  const randn = () => (rand() - 0.5) * 0.4;

  const params = {
    embedding: Array.from({ length: vocabSize }, () =>
      Float32Array.from({ length: embedDim }, randn)
    ),
    blocks: [],
    Wout: Array.from({ length: vocabSize }, () =>
      Float32Array.from({ length: embedDim }, randn)
    ),
    bout: new Float32Array(vocabSize),
  };

  for (let b = 0; b < numBlocks; b++) {
    params.blocks.push({
      Wq: Array.from({ length: embedDim }, () => Float32Array.from({ length: embedDim }, randn)),
      Wk: Array.from({ length: embedDim }, () => Float32Array.from({ length: embedDim }, randn)),
      Wv: Array.from({ length: embedDim }, () => Float32Array.from({ length: embedDim }, randn)),
      Wo: Array.from({ length: embedDim }, () => Float32Array.from({ length: embedDim }, randn)),
      bq: new Float32Array(embedDim),
      bk: new Float32Array(embedDim),
      bv: new Float32Array(embedDim),
      bo: new Float32Array(embedDim),
      ln1_g: Float32Array.from({ length: embedDim }, () => 1),
      W_gate: Array.from({ length: ffnDim }, () => Float32Array.from({ length: embedDim }, randn)),
      b_gate: new Float32Array(ffnDim),
      W1: Array.from({ length: ffnDim }, () => Float32Array.from({ length: embedDim }, randn)),
      b1: new Float32Array(ffnDim),
      W2: Array.from({ length: embedDim }, () => Float32Array.from({ length: ffnDim }, randn)),
      b2: new Float32Array(embedDim),
      ln2_g: Float32Array.from({ length: embedDim }, () => 1),
    });
  }

  return params;
}

export function forward(params, tokens, config) {
  const { embedDim, numHeads, numBlocks = 1 } = config;
  const headDim = embedDim / numHeads;
  const len = tokens.length;

  let hidden = tokens.map((t) => params.embedding[t].slice());

  const freqs = precomputeFreqs(len, headDim);
  const allBlockAttnWeights = [];

  for (let blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
    const block = params.blocks[blockIdx];

    const normed = hidden.map((h) => rmsNorm(h, block.ln1_g));
    const Q = normed.map((h) => vecAdd(matVec(block.Wq, h), block.bq));
    const K = normed.map((h) => vecAdd(matVec(block.Wk, h), block.bk));
    const V = normed.map((h) => vecAdd(matVec(block.Wv, h), block.bv));

    // Apply RoPE to Q and K per-head
    for (let head = 0; head < numHeads; head++) {
      const off = head * headDim;
      for (let pos = 0; pos < len; pos++) {
        const qSlice = Q[pos].slice(off, off + headDim);
        const kSlice = K[pos].slice(off, off + headDim);
        Q[pos].set(applyRoPE(qSlice, pos, freqs), off);
        K[pos].set(applyRoPE(kSlice, pos, freqs), off);
      }
    }

    let attnOutputs = Array.from({ length: len }, () => new Float32Array(embedDim));
    let blockAttnWeights = [];

    for (let head = 0; head < numHeads; head++) {
      const offset = head * headDim;
      const headWeights = [];

      for (let i = 0; i < len; i++) {
        const scores = [];
        for (let j = 0; j <= i; j++) {
          let score = 0;
          for (let d = 0; d < headDim; d++) {
            score += Q[i][offset + d] * K[j][offset + d];
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
            val += weights[j] * V[j][offset + d];
          }
          attnOutputs[i][offset + d] = val;
        }
      }
      blockAttnWeights.push(headWeights);
    }
    allBlockAttnWeights.push(blockAttnWeights);

    const attnProj = attnOutputs.map((h) => vecAdd(matVec(block.Wo, h), block.bo));
    hidden = hidden.map((h, i) => vecAdd(h, attnProj[i]));

    const normed2 = hidden.map((h) => rmsNorm(h, block.ln2_g));
    // GLU FFN: gate = activation(W_gate @ x), up = W1 @ x, out = W2 @ (gate * up)
    const activationFn = { silu, gelu, relu }[config.activation] || silu;
    const gate = normed2.map((h) => activationFn(vecAdd(matVec(block.W_gate, h), block.b_gate)));
    const up = normed2.map((h) => vecAdd(matVec(block.W1, h), block.b1));
    const gated = gate.map((g, i) => vecMul(g, up[i]));
    const ffn2 = gated.map((h) => vecAdd(matVec(block.W2, h), block.b2));
    hidden = hidden.map((h, i) => vecAdd(h, ffn2[i]));
  }

  const lastHidden = hidden[hidden.length - 1];
  const logits = vecAdd(matVec(params.Wout, lastHidden), params.bout);

  return { logits, attnWeights: allBlockAttnWeights, embeddings: params.embedding };
}

export function countParameters(params) {
  let count = 0;
  function countArray(arr) {
    if (arr[0]?.length !== undefined) {
      arr.forEach((row) => (count += row.length));
    } else {
      count += arr.length;
    }
  }
  countArray(params.embedding);
  countArray(params.Wout);
  countArray(params.bout);
  for (const block of params.blocks) {
    for (const key of Object.keys(block)) {
      countArray(block[key]);
    }
  }
  return count;
}

import { mulberry32, matVec, vecAdd, layerNorm, softmax, relu } from "./math.js";

export function createTinyTransformer(config) {
  const { vocabSize, embedDim, ffnDim, seqLen, numBlocks = 1 } = config;
  const rand = mulberry32(42);
  const randn = () => (rand() - 0.5) * 0.4;

  const params = {
    embedding: Array.from({ length: vocabSize }, () =>
      Float32Array.from({ length: embedDim }, randn)
    ),
    posEmbedding: Array.from({ length: seqLen }, () =>
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
      ln1_b: new Float32Array(embedDim),
      W1: Array.from({ length: ffnDim }, () => Float32Array.from({ length: embedDim }, randn)),
      b1: new Float32Array(ffnDim),
      W2: Array.from({ length: embedDim }, () => Float32Array.from({ length: ffnDim }, randn)),
      b2: new Float32Array(embedDim),
      ln2_g: Float32Array.from({ length: embedDim }, () => 1),
      ln2_b: new Float32Array(embedDim),
    });
  }

  return params;
}

export function forward(params, tokens, config) {
  const { embedDim, numHeads, numBlocks = 1 } = config;
  const headDim = embedDim / numHeads;
  const len = tokens.length;

  let hidden = tokens.map((t, pos) =>
    vecAdd(params.embedding[t], params.posEmbedding[pos])
  );

  const allBlockAttnWeights = [];

  for (let blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
    const block = params.blocks[blockIdx];

    const normed = hidden.map((h) => layerNorm(h, block.ln1_g, block.ln1_b));
    const Q = normed.map((h) => vecAdd(matVec(block.Wq, h), block.bq));
    const K = normed.map((h) => vecAdd(matVec(block.Wk, h), block.bk));
    const V = normed.map((h) => vecAdd(matVec(block.Wv, h), block.bv));

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

    const normed2 = hidden.map((h) => layerNorm(h, block.ln2_g, block.ln2_b));
    const ffn1 = normed2.map((h) => relu(vecAdd(matVec(block.W1, h), block.b1)));
    const ffn2 = ffn1.map((h) => vecAdd(matVec(block.W2, h), block.b2));
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
  countArray(params.posEmbedding);
  countArray(params.Wout);
  countArray(params.bout);
  for (const block of params.blocks) {
    for (const key of Object.keys(block)) {
      countArray(block[key]);
    }
  }
  return count;
}

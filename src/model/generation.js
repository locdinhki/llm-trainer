// KV Cache for efficient autoregressive generation.
// Instead of recomputing all positions each step, we cache K/V vectors
// and only compute the new token's Q, K, V — then attend to the full cache.

import { matVec, vecAdd, vecMul, rmsNorm, softmax, silu } from "./math.js";
import { precomputeFreqs, applyRoPE } from "./rope.js";

export function createKVCache(config) {
  const numBlocks = config.numBlocks || 1;
  return {
    blocks: Array.from({ length: numBlocks }, () => ({
      K: [], // Array of Float32Array[embedDim], one per cached position
      V: [], // Array of Float32Array[embedDim], one per cached position
    })),
    length: 0, // number of positions cached so far
  };
}

// Single-token forward pass using KV cache.
// Processes one token at the given position, appends its K/V to the cache,
// and returns logits for next-token prediction.
// Returns { logits, kvCache } with the updated cache.
export function forwardWithKVCache(params, token, position, kvCache, config) {
  const { embedDim, numHeads, numBlocks = 1 } = config;
  const headDim = embedDim / numHeads;

  // We need freqs up to position+1 since position is 0-indexed
  const freqs = precomputeFreqs(position + 1, headDim);

  // Embed the single token
  let hidden = params.embedding[token].slice();

  for (let blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
    const block = params.blocks[blockIdx];
    const cache = kvCache.blocks[blockIdx];

    // RMSNorm 1
    const normed1 = rmsNorm(hidden, block.ln1_g);

    // Q, K, V projections for this single position
    const q = vecAdd(matVec(block.Wq, normed1), block.bq);
    const k = vecAdd(matVec(block.Wk, normed1), block.bk);
    const v = vecAdd(matVec(block.Wv, normed1), block.bv);

    // Apply RoPE to Q and K per-head
    for (let head = 0; head < numHeads; head++) {
      const off = head * headDim;
      const qSlice = q.slice(off, off + headDim);
      const kSlice = k.slice(off, off + headDim);
      q.set(applyRoPE(qSlice, position, freqs), off);
      k.set(applyRoPE(kSlice, position, freqs), off);
    }

    // Append K, V to cache
    cache.K.push(k);
    cache.V.push(v);

    // Attend: current Q attends to all cached K/V (positions 0..position)
    const attnOutput = new Float32Array(embedDim);
    const cachedLen = cache.K.length; // = position + 1

    for (let head = 0; head < numHeads; head++) {
      const offset = head * headDim;

      // Compute attention scores: Q[current] dot K[j] for all cached positions
      const scores = new Float32Array(cachedLen);
      const scale = 1 / Math.sqrt(headDim);
      for (let j = 0; j < cachedLen; j++) {
        let score = 0;
        for (let d = 0; d < headDim; d++) {
          score += q[offset + d] * cache.K[j][offset + d];
        }
        scores[j] = score * scale;
      }

      // Softmax (causal mask not needed — we only have positions 0..current)
      const weights = softmax(scores);

      // Weighted sum of V
      for (let d = 0; d < headDim; d++) {
        let val = 0;
        for (let j = 0; j < cachedLen; j++) {
          val += weights[j] * cache.V[j][offset + d];
        }
        attnOutput[offset + d] = val;
      }
    }

    // Wo projection + residual
    const attnProj = vecAdd(matVec(block.Wo, attnOutput), block.bo);
    hidden = vecAdd(hidden, attnProj);

    // RMSNorm 2
    const normed2 = rmsNorm(hidden, block.ln2_g);

    // SwiGLU FFN
    const gatePre = vecAdd(matVec(block.W_gate, normed2), block.b_gate);
    const gate = silu(gatePre);
    const up = vecAdd(matVec(block.W1, normed2), block.b1);
    const gated = vecMul(gate, up);
    const ffn2 = vecAdd(matVec(block.W2, gated), block.b2);

    // Second residual
    hidden = vecAdd(hidden, ffn2);
  }

  // Output projection
  const logits = vecAdd(matVec(params.Wout, hidden), params.bout);

  kvCache.length = position + 1;

  return { logits, kvCache };
}

// Generate tokens autoregressively using KV cache.
// Takes a prompt (array of token ids) and generates up to maxNewTokens.
// Returns the full sequence (prompt + generated tokens).
export function generate(params, promptTokens, config, maxNewTokens, temperature = 1.0) {
  const kvCache = createKVCache(config);
  const tokens = [...promptTokens];

  // Process prompt tokens through cache (we only need the last logits)
  let logits;
  for (let i = 0; i < promptTokens.length; i++) {
    ({ logits } = forwardWithKVCache(params, promptTokens[i], i, kvCache, config));
  }

  // Generate new tokens
  for (let step = 0; step < maxNewTokens; step++) {
    // Sample from logits
    const probs = softmax(
      temperature === 1.0
        ? logits
        : logits.map((l) => l / temperature)
    );

    // Weighted random sampling
    let r = Math.random();
    let nextToken = probs.length - 1;
    for (let i = 0; i < probs.length; i++) {
      r -= probs[i];
      if (r <= 0) {
        nextToken = i;
        break;
      }
    }

    tokens.push(nextToken);

    // Forward the new token
    const position = promptTokens.length + step;
    ({ logits } = forwardWithKVCache(params, nextToken, position, kvCache, config));
  }

  return tokens;
}

// Sample a single token from logits with temperature, top-k, and top-p filtering.
// Returns { tokenId, probability } where probability is the post-temperature softmax value.
export function sampleToken(logits, temperature, topK, topP) {
  const n = logits.length;

  // Temperature scaling
  const scaled = new Float32Array(n);
  for (let i = 0; i < n; i++) scaled[i] = logits[i] / temperature;

  // Softmax
  const probs = softmax(scaled);

  // Build indexed pairs and sort descending by probability
  const indexed = new Array(n);
  for (let i = 0; i < n; i++) indexed[i] = { id: i, prob: probs[i] };
  indexed.sort((a, b) => b.prob - a.prob);

  // Top-k: keep only the top-k entries
  const topKFiltered = indexed.slice(0, topK);

  // Top-p (nucleus): accumulate until sum >= topP
  let cumSum = 0;
  let nucleusEnd = 0;
  for (let i = 0; i < topKFiltered.length; i++) {
    cumSum += topKFiltered[i].prob;
    nucleusEnd = i + 1;
    if (cumSum >= topP) break;
  }
  const candidates = topKFiltered.slice(0, nucleusEnd);

  // Renormalize
  let total = 0;
  for (const c of candidates) total += c.prob;
  for (const c of candidates) c.prob /= total;

  // Sample from the filtered distribution
  let r = Math.random();
  let chosen = candidates[candidates.length - 1];
  for (const c of candidates) {
    r -= c.prob;
    if (r <= 0) { chosen = c; break; }
  }

  return { tokenId: chosen.id, probability: probs[chosen.id] };
}

// Streaming autoregressive generation using KV cache.
// Async generator that yields tokens one at a time for visual drip-feed.
// Yields { type: "prompt", tokens } then { type: "generated", token } per step.
export async function* generateStream(
  params, config, tokenizer, inputText,
  maxTokens, temperature, topK, topP, signal
) {
  const promptTokens = tokenizer.encode(inputText);
  if (promptTokens.length === 0) return;

  const kvCache = createKVCache(config);

  // Prefill: run forward for all prompt tokens to populate KV cache
  let logits;
  for (let i = 0; i < promptTokens.length; i++) {
    ({ logits } = forwardWithKVCache(params, promptTokens[i], i, kvCache, config));
  }

  // Yield prompt tokens (probability = 1.0 since they are user-provided)
  yield {
    type: "prompt",
    tokens: promptTokens.map((id) => ({
      id,
      text: tokenizer.id2token[id],
      probability: 1.0,
    })),
  };

  // Autoregressive generation loop
  for (let step = 0; step < maxTokens; step++) {
    if (signal && signal.aborted) return;

    const { tokenId, probability } = sampleToken(logits, temperature, topK, topP);

    yield {
      type: "generated",
      token: {
        id: tokenId,
        text: tokenizer.id2token[tokenId],
        probability,
      },
    };

    // Early stopping: if the model is too uncertain
    const fullProbs = softmax(logits);
    let maxProb = 0;
    for (let i = 0; i < fullProbs.length; i++) {
      if (fullProbs[i] > maxProb) maxProb = fullProbs[i];
    }
    if (maxProb < 0.05) return;

    // Forward the new token through the cache
    const position = promptTokens.length + step;
    ({ logits } = forwardWithKVCache(params, tokenId, position, kvCache, config));

    // Delay for visual effect
    await new Promise((r) => setTimeout(r, 100));
  }
}

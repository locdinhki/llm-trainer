# 2E: KV Cache for Efficient Generation

## Problem
During autoregressive generation, the model generates one token at a time. Without caching, generating token N requires recomputing K and V for ALL N-1 previous tokens — O(N^2) total work for a full sequence.

## How KV Cache Works
Store the K and V projections from previous tokens. For each new token:
1. Compute Q, K, V for the **single new token only**
2. Append new K, V to the cache
3. Compute attention: new Q attends to ALL cached K/V
4. Return logits for next token prediction

This makes each generation step O(N) instead of O(N^2) total.

## Cache Structure
```js
kvCache = {
  blocks: [
    {
      K: [],  // Array of Float32Array, one per cached position
      V: [],  // Array of Float32Array, one per cached position
    },
    // ... one per transformer block
  ]
}
```

## Implementation

### New file: `src/model/generation.js`

```js
// Create empty KV cache
createKVCache(config) -> kvCache

// Forward pass for a single token, using and updating the cache
forwardWithKVCache(params, token, position, kvCache, config):
  // 1. Embed the single token (no position embedding — RoPE handles it)
  // 2. For each block:
  //    a. RMSNorm
  //    b. Compute Q, K, V for this single token
  //    c. Apply RoPE to Q, K at position
  //    d. Append K, V to cache
  //    e. Attention: Q attends to all cached K/V (causal mask is implicit — only past tokens)
  //    f. SwiGLU FFN
  //    g. Residual connections
  // 3. Output projection -> logits
  return { logits, kvCache }
```

## When It's Used
- Phase 7 (Generation Panel): Used during autoregressive text generation
- NOT used during training (training uses full forwardWithCache for all positions)
- NOT used for visualization (those need full sequence attention weights)

## Performance Impact
For generating a 32-token sequence:
- Without KV cache: 32 * 32/2 = 512 attention computations (average)
- With KV cache: 32 * 16 = 512 attention computations (same asymptotically, but avoids redundant Q/K/V projections for previous tokens)

The real win is avoiding ~31 redundant Q/K/V projection matrix multiplies per generation step.

## Notes
- KV cache is inference-only — training always uses the full forward/backward pass
- Cache must be reset when starting a new generation
- Cache entries are Float32Array for consistency with Phase 1

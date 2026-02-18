# 2A: RoPE (Rotary Position Embeddings)

## What It Replaces
Absolute learned positional embeddings (`posEmbedding` matrix added to token embeddings).

## How RoPE Works
Instead of adding position information to the input, RoPE applies a rotation to Q and K vectors based on their position. The rotation angle depends on both the position and the dimension index.

For each pair of dimensions (2i, 2i+1) in a head:
```
theta = position * base_freq^(-2i / headDim)

q_rot[2i]   = q[2i] * cos(theta) - q[2i+1] * sin(theta)
q_rot[2i+1] = q[2i] * sin(theta) + q[2i+1] * cos(theta)
```

The same rotation is applied to K. When Q and K are dot-producted, the result naturally encodes the *relative* position between tokens (not absolute), which generalizes better.

## Why It's Better
- **Relative position**: Attention scores depend on distance between tokens, not absolute position
- **Extrapolation**: Can handle sequences longer than training length (to some extent)
- **No learned parameters**: RoPE is purely mathematical (sin/cos), reducing param count
- **Industry standard**: Used by LLaMA, LLaMA 2/3, Mistral, Qwen, Phi, CodeLlama

## New File: `src/model/rope.js`

```js
// Precompute sin/cos frequency tables
precomputeFreqs(seqLen, headDim, base = 10000)
  -> { cos: Float32Array, sin: Float32Array }

// Apply rotation to a vector at a given position
applyRoPE(vec, position, freqs)
  -> rotated Float32Array

// Backward pass: inverse rotation (negate sin terms)
applyRoPEBackward(dVec, position, freqs)
  -> rotated Float32Array
```

## Changes to Existing Files

### `transformer.js`
- Remove `posEmbedding` from `createTinyTransformer()`
- In `forward()`: remove `vecAdd(embedding, posEmbedding)`, just use `embedding[t]`
- After Q/K projection, apply `applyRoPE(Q[pos], pos, freqs)` per-head
- Precompute `freqs` once at start of forward pass

### `forwardWithCache.js`
- Same changes as transformer.js
- Cache pre-rotation Q/K (needed for backward RoPE)

### `backward.js`
- Apply `applyRoPEBackward` when computing dQ and dK
- Remove `dPosEmbedding` from `createZeroGrads()`
- Remove posEmbedding gradient accumulation from `backward()`

### `training.js`
- `verifyGradients` no longer checks posEmbedding gradients

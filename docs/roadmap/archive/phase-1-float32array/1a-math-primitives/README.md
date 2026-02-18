# 1A: Math Primitives — Float32Array Migration

## File
`src/model/math.js`

## Changes
- All functions that create output arrays (`matVec`, `vecAdd`, `softmax`, `relu`, `layerNorm`) return `Float32Array` instead of `new Array()`
- `vecAddInPlace` already uses explicit loops — just ensure it works with Float32Array
- Add `gelu(x)` and `silu(x)` activation functions
- Add `geluBackward(x, dOut)` and `siluBackward(x, dOut)` for backward pass
- Keep `pca2D` using regular Arrays (returns `{x, y}` objects, not perf-critical)

## Implementation Pattern
```js
// Before
export function matVec(mat, vec) {
  const out = new Array(mat.length);
  for (let i = 0; i < mat.length; i++) { ... }
  return out;
}

// After
export function matVec(mat, vec) {
  const out = new Float32Array(mat.length);
  for (let i = 0; i < mat.length; i++) { ... }
  return out;
}
```

## New Functions
```js
// SiLU (Sigmoid Linear Unit / Swish) — used by LLaMA
silu(x): Float32Array  // x[i] * sigmoid(x[i])

// GELU (Gaussian Error Linear Unit) — used by GPT-2
gelu(x): Float32Array  // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

// Backward variants
siluBackward(x, dOut): Float32Array
geluBackward(x, dOut): Float32Array
```

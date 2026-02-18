# 2B: RMSNorm (Root Mean Square Normalization)

## What It Replaces
LayerNorm, which normalizes by both mean and variance:
```
LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
```

## How RMSNorm Works
RMSNorm only normalizes by the root-mean-square — no mean subtraction, no beta parameter:
```js
rmsNorm(x, gamma, eps = 1e-6):
  rms = sqrt(mean(x^2) + eps)
  return gamma * (x / rms)
```

## Why It's Better
- **Simpler**: One fewer statistic to compute (no mean subtraction)
- **Faster**: Single pass over the data instead of two
- **Fewer parameters**: No beta (shift) parameter — only gamma (scale)
- **Empirically equivalent**: Same quality as LayerNorm in practice
- **Industry standard**: Used by LLaMA, LLaMA 2/3, Mistral, Gemma

## Implementation

### `src/model/math.js` — Add `rmsNorm(x, gamma)`
```js
export function rmsNorm(x, gamma) {
  const n = x.length;
  let sumSq = 0;
  for (let i = 0; i < n; i++) sumSq += x[i] * x[i];
  const rms = Math.sqrt(sumSq / n + 1e-6);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = gamma[i] * (x[i] / rms);
  return out;
}
```

### `src/model/backward.js` — Add `rmsNormBackward(x, gamma, dOut)`
Returns `{ dX, dGamma }` (no dBeta since there's no beta parameter).

### `src/model/transformer.js` — Update block params
- Remove `ln1_b` and `ln2_b` from each block
- Replace `layerNorm(h, g, b)` calls with `rmsNorm(h, g)`
- Keep `ln1_g` and `ln2_g` (gamma/scale parameters)

### `src/model/backward.js` — Update backward
- Replace `layerNormBackward` with `rmsNormBackward`
- Remove dBeta accumulation from gradient computation

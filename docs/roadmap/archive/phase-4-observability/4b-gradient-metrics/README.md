# 4B: Gradient & Metrics Tracking

## Files
- `src/model/training.js`
- `src/App.jsx`

## Gradient Norm

### `computeGradNorm(grads)`
Compute L2 norm of the full gradient vector. Follows the same iteration pattern as `scaleGrads`:
```js
sumSq += g[i] * g[i]  // for every gradient element
return Math.sqrt(sumSq)
```

### Return Type Change
`trainStepMiniBatch` returns `{ loss, gradNorm }` instead of a scalar loss. Gradient norm is computed after scaling but before applying the optimizer.

### App.jsx Buffering
- `gradNormBufferRef` accumulates norms between render frames (same pattern as `lossBufferRef`)
- `gradNormHistory` state (capped at 500 entries)
- Flushed in render loop, reset on model reinit

## Perplexity
Derived from loss at render time — no separate buffer needed:
```js
const perplexityHistory = useMemo(() => lossHistory.map(l => Math.exp(l)), [lossHistory]);
```
- Random baseline: `vocabSize` (e.g., perplexity 10 for 10-word vocab)
- Perfect model: perplexity 1

## Learning Rate History
- `lrBufferRef` records actual LR used each step
- `lrHistory` state (capped at 500 entries)
- Shows the real LR trajectory including warmup and cosine decay

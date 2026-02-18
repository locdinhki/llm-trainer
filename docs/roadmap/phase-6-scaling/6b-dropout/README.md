# 6B: Dropout

## Files
- `src/model/transformer.js`
- `src/model/forwardWithCache.js`
- `src/model/backward.js`

## What Is Dropout?
Randomly zeroes out a fraction of activations during training. This forces the model to learn redundant representations, preventing overfitting.

```
dropout(x, rate):
  mask = random(0, 1) > rate for each element
  return x * mask / (1 - rate)   // scale to maintain expected value
```

## Where to Apply
1. After attention output projection (before residual add)
2. After FFN output (before residual add)

## Config
- `config.dropout`: float 0.0 to 0.3 (default: 0.0 for small models, 0.1 for medium+)
- `config.training`: boolean flag (true during training, false during inference/visualization)

## Forward Pass
```js
// In training mode
if (config.dropout > 0 && config.training) {
  const mask = generateDropoutMask(attnProj.length, config.dropout);
  applyDropoutInPlace(attnProj, mask, config.dropout);
  // Store mask in cache for backward pass
}
```

## Backward Pass
Dropout backward: multiply dOut by the same mask used in forward (and scale by 1/(1-rate)).

## Inference
Dropout is DISABLED during:
- Visualization updates (forward pass for selected prompt)
- Generation (autoregressive token generation)
- Gradient verification

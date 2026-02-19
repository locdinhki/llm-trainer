# 3B: Adam Optimizer

## File
`src/model/training.js`

## What Is Adam?
Adam (Adaptive Moment Estimation) maintains per-parameter running averages of:
- **m** (first moment / momentum): exponential moving average of gradients
- **v** (second moment): exponential moving average of squared gradients

Update rule:
```
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad^2
m_hat = m / (1 - beta1^t)     // bias correction
v_hat = v / (1 - beta2^t)     // bias correction
param -= lr * m_hat / (sqrt(v_hat) + eps)
```

## Implementation

### `createAdamState(params, config)`
Creates optimizer state matching the params structure:
```js
{
  m: createZeroGrads(params, config),  // first moment (same shape as params)
  v: createZeroGrads(params, config),  // second moment (same shape as params)
  t: 0,                                 // step counter for bias correction
}
```

### `adamUpdate(params, grads, state, lr, beta1=0.9, beta2=0.999, eps=1e-8)`
Walks both params and grads in parallel, applies Adam update in-place.

## Memory Impact
Adam requires 2x the parameter storage (m + v). For ~87K params at Float32:
- Params: ~340KB
- Adam state: ~680KB
- Total: ~1MB — trivial for browser

## Hyperparameters
| Param | Default | Description |
|-------|---------|-------------|
| lr | 0.001 | Base learning rate (lower than SGD's 0.01) |
| beta1 | 0.9 | Momentum decay |
| beta2 | 0.999 | RMS decay |
| eps | 1e-8 | Numerical stability |

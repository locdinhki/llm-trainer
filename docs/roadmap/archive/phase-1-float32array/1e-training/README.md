# 1E: Training — Float32Array

## File
`src/model/training.js`

## Changes

### `scaleGrads(grads, scale)`
- Already uses explicit for-loops — works with Float32Array as-is
- Verify it handles Array-of-Float32Array 2D pattern

### `applyGradients(params, grads, lr, maxNorm)`
- Element-wise gradient clipping already uses explicit loops
- Works with Float32Array without changes

### `trainStepBackprop(params, trainingData, config, lr)`
- No changes needed — calls `forwardWithCache` and `backward` which handle the typed arrays
- Loss computation uses regular JS numbers (not arrays)

### `verifyGradients(params, trainingData, config)`
- Numerical gradient perturbation works the same on Float32Array values
- May need to adjust array access pattern if 2D matrices change

## Notes
- Float32Array has ~7 decimal digits of precision (vs ~15 for JS number/Float64)
- This may slightly affect gradient verification thresholds
- Gradient check tolerance might need to be relaxed from 1e-4 to 1e-3 for Float32

# 3A: Mini-Batch Training

## File
`src/model/training.js`

## Implementation

### `trainStepMiniBatch(params, trainingData, config, lr, batchSize, optimState)`

```js
1. Randomly sample `batchSize` indices from trainingData
2. Initialize zero gradients
3. For each sampled example:
   a. forwardWithCache -> cache
   b. backward -> accumulate gradients
4. Average gradients: scaleGrads(grads, 1 / batchSize)
5. Apply optimizer update (SGD or Adam)
6. Return average loss over the batch
```

## Key Details
- Default `batchSize = 32`
- If trainingData has fewer than batchSize examples, use all of them (full-batch fallback)
- Random sampling uses `Math.random()` — no need for a seeded RNG here
- Track epoch count: one epoch = `ceil(trainingData.length / batchSize)` steps

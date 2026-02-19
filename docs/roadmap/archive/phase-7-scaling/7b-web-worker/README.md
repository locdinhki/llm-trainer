# 6C: Web Worker Integration

## Files
- `src/model/training.worker.js` (already exists, needs update)
- `src/model/useTrainingWorker.js` (already exists, needs update)
- `src/App.jsx`

## Current State
Worker infrastructure was built during the optimization phase but NOT wired into the training loop. App.jsx calls `trainStepBackprop` directly on the main thread.

## Changes

### `training.worker.js` — Update message handlers
- `init`: Create model, tokenizer, training data, Adam optimizer state
- `trainBatch`: Run N mini-batch steps, return updated params + losses
- `forward`: Run forward pass for visualization (single prompt)
- `setConfig`: Update config without full reinit

### `useTrainingWorker.js` — Update hook
- `initWorker(config, sentences, tokenizerConfig)`: Initialize everything in worker
- `train(lr, steps, batchSize)`: Run training steps, return params + losses
- `forwardPass(tokens)`: Forward pass for visualization
- `supported`: Boolean flag for Worker availability

### `App.jsx` — Wire up training loop
```
Training loop (main thread):
  1. Post "trainBatch" message to worker
  2. Worker runs N steps, posts back { params, losses }
  3. Main thread updates paramsRef, buffers losses

Render loop (main thread, rAF):
  1. Run forward pass on main thread with latest params (for viz)
  2. Flush loss/log buffers to React state
```

### Data Transfer
- Use `Transferable` for Float32Array buffers to avoid copy overhead
- Worker keeps its own copy of params/optimizer state
- Only send params back periodically (every 10 steps) for visualization
- Send losses back every step (small data, no transfer overhead)

### Fallback
If `typeof Worker === 'undefined'` or worker creation fails:
- Use main-thread training (same as current)
- No functionality loss
- `supported` flag in hook returns false

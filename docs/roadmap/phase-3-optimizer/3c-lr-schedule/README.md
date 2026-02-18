# 3C: Learning Rate Schedule

## File
`src/model/training.js`

## Why Scheduling Matters
- **Too high LR at start**: Unstable gradients, loss spikes, divergence
- **Constant LR**: Overshoots minimum, oscillates instead of converging
- **Decaying LR**: Large steps early (fast progress), small steps late (fine-tuning)

## Implementation

### Warmup + Cosine Decay

```js
getLearningRate(step, warmupSteps, totalSteps, baseLR):
  if (step < warmupSteps):
    // Linear warmup: 0 -> baseLR
    return baseLR * (step / warmupSteps)
  else:
    // Cosine decay: baseLR -> 0
    progress = (step - warmupSteps) / (totalSteps - warmupSteps)
    return baseLR * 0.5 * (1 + cos(pi * progress))
```

### Default Schedule
- **Warmup**: 100 steps (linear ramp from 0 to baseLR)
- **Total steps**: configurable, default ~5,000
- **Base LR**: 0.001 (for Adam)

### Visualization
The LR schedule could be shown as a small sparkline in the training log or loss panel header.

## Used By
Almost every modern LLM training run uses warmup + cosine decay:
- GPT-3, LLaMA, Mistral, Gemma, etc.

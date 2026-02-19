# 4A: AdamW Optimizer

## File: `src/model/training.js`

## What Is AdamW?
Standard Adam applies weight decay inside the gradient update (equivalent to L2 regularization). AdamW decouples weight decay from the adaptive learning rate, applying it directly to the parameters:

```
Adam:  p -= lr * m_hat / (sqrt(v_hat) + eps)           // weight decay buried in gradients
AdamW: p -= lr * (m_hat / (sqrt(v_hat) + eps) + λ * p) // weight decay applied separately
```

This is the standard optimizer for GPT-3, LLaMA, PaLM, and most modern LLMs.

## Changes

### `adamUpdateArray1D` / `adamUpdateArray2D`
Add `weightDecay` parameter. After computing the Adam step, apply decoupled weight decay:
```js
p[i] -= lr * (mHat / (Math.sqrt(vHat) + eps) + weightDecay * p[i]);
```

### `adamUpdate`
Add `weightDecay` parameter (default 0.01), pass through to array update functions.

### `trainStepMiniBatch`
Add `weightDecay` parameter, pass through to `adamUpdate`.

### Settings UI
- Weight decay slider: 0 to 0.1, step 0.001, default 0.01
- Rename optimizer label from "Adam" to "AdamW"

# Phase 3: Mini-Batch SGD + Adam Optimizer

## Goal
Replace full-batch SGD with mini-batch training and the Adam optimizer. Essential for scaling to larger datasets where processing all training examples per step is too slow.

## Why This Matters
- **Full-batch** (current): processes all ~28 training pairs per step. At 600+ pairs (expanded corpus), each step takes too long.
- **Mini-batch**: randomly samples 32 examples per step. Faster per step, introduces beneficial noise (stochastic gradient).
- **Adam**: adapts learning rate per-parameter using momentum + RMS of gradients. Converges much faster than plain SGD at larger scales.

## Subphases

| # | Subphase | Description |
|---|----------|-------------|
| 3A | [Mini-Batch](3a-mini-batch/) | Random sampling of training examples per step |
| 3B | [Adam Optimizer](3b-adam/) | Adaptive learning rate with momentum |
| 3C | [LR Schedule](3c-lr-schedule/) | Warmup + cosine decay for stable training |

## Files
- `src/model/training.js`

## Verification
- [ ] Mini-batch training converges (may be noisier than full-batch)
- [ ] Adam converges faster than SGD on same data
- [ ] LR warmup prevents early training instability
- [ ] Gradient check still passes

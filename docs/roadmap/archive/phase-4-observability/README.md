# Phase 4: Training Observability + Checkpoints

## Goal
Add industry-standard training diagnostics — AdamW optimizer, gradient norm tracking, perplexity metric, learning rate visualization — and checkpoint save/load. These tools let users observe and debug training the way real ML engineers do.

## Subphases

| # | Subphase | Description |
|---|----------|-------------|
| 4A | [AdamW Optimizer](4a-adamw/) | Decoupled weight decay upgrade from Adam |
| 4B | [Gradient & Metrics](4b-gradient-metrics/) | Gradient norm tracking, perplexity, LR history |
| 4C | [Metrics Panel](4c-metrics-panel/) | Visualization panel for perplexity, gradient norm, LR curve |
| 4D | [Checkpoints](4d-checkpoints/) | Save/load model state as JSON files |

## Files
- `src/model/training.js` (modify)
- `src/model/checkpoint.js` (new)
- `src/components/MetricsPanel.jsx` (new)
- `src/components/SettingsPanel.jsx` (modify)
- `src/App.jsx` (modify)

## Verification
- [x] AdamW converges with weight decay enabled
- [x] Gradient norm history plotted, shows clipping events
- [x] Perplexity drops from ~vocabSize toward 1 during training
- [x] LR curve matches warmup + cosine schedule
- [x] Checkpoint save → reset → load restores training state
- [x] Checkpoint with mismatched config shows error

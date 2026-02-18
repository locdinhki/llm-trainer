# Phase 6: Model Scaling + Architecture Options

## Goal
Expand configurable options to support larger models (up to ~300K params), add dropout for regularization, and wire up the existing Web Worker for off-main-thread training.

## Subphases

| # | Subphase | Description |
|---|----------|-------------|
| 6A | [Config Options](6a-config-options/) | Expanded settings with preset configs |
| 6B | [Dropout](6b-dropout/) | Regularization for larger models |
| 6C | [Web Worker](6c-web-worker/) | Move training off main thread |

## Files
- `src/components/SettingsPanel.jsx`
- `src/model/transformer.js`
- `src/model/forwardWithCache.js`
- `src/model/backward.js`
- `src/model/training.worker.js`
- `src/model/useTrainingWorker.js`
- `src/App.jsx`

## Verification
- [ ] All preset configs create valid models
- [ ] Param count display matches actual count
- [ ] Dropout reduces overfitting on small corpus
- [ ] Web Worker: training runs off main thread
- [ ] UI stays at 60fps during max-speed training with Web Worker
- [ ] Fallback to main thread if Worker unavailable

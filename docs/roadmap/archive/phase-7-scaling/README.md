# Phase 7: Model Scaling + Architecture Options

## Goal
Expand configurable options to support larger models (up to ~300K params) and wire up the existing Web Worker for off-main-thread training.

## Subphases

| # | Subphase | Description |
|---|----------|-------------|
| 7A | [Config Options](7a-config-options/) | Expanded settings with preset configs |
| 7B | [Web Worker](7b-web-worker/) | Move training off main thread |

## Files
- `src/components/SettingsPanel.jsx`
- `src/model/transformer.js`
- `src/model/forwardWithCache.js`
- `src/model/backward.js`
- `src/model/training.worker.js`
- `src/model/useTrainingWorker.js`
- `src/App.jsx`

## Verification
- [x] All preset configs create valid models
- [x] Param count display matches actual count
- [x] Web Worker: training runs off main thread
- [x] UI stays at 60fps during max-speed training with Web Worker
- [x] Fallback to main thread if Worker unavailable

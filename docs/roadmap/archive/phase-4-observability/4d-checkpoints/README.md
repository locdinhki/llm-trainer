# 4D: Checkpoint Save/Load

## New File: `src/model/checkpoint.js`

## Serialization Challenge
Parameters use `Float32Array` which `JSON.stringify` converts to `{"0": 0.1, "1": 0.2, ...}` — inefficient and loses type info. Solution: base64-encode the raw binary.

## Functions

### `float32ToBase64(f32arr)` / `base64ToFloat32(base64)`
Convert between Float32Array and base64 strings using `btoa`/`atob`.

### `serializeParams(params)` / `deserializeParams(data)`
Recursively serialize the nested param structure (embedding, Wout, bout, blocks[*]). Follows the same key iteration pattern as `scaleGrads` in training.js. Works for both model params and Adam state (same structure).

### `saveCheckpoint(params, adamState, step, config, lossHistory)`
Returns JSON string:
```json
{
  "version": 1,
  "config": { "vocabSize": 10, "embedDim": 16, ... },
  "step": 250,
  "lossHistory": [2.3, 2.1, ...],
  "params": { "embedding": ["base64..."], ... },
  "adamState": { "m": {...}, "v": {...}, "t": 250 }
}
```

### `loadCheckpoint(jsonString)`
Parse and reconstruct. Returns `{ config, step, lossHistory, params, adamState }`.

## UI Integration
- **Save**: Blob download as `checkpoint-step-N.json` (~30-40KB for ~2,700 params)
- **Load**: Hidden file input, parse, validate config.vocabSize match, restore all refs + state
- **Buttons**: In SettingsPanel under "Checkpoints" section
- **Logging**: Log save/load events to training log

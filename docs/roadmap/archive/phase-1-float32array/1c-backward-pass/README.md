# 1C: Backward Pass — Float32Array

## File
`src/model/backward.js`

## Changes

### Helper functions
```js
// Before
function zeros1D(n) { return new Array(n).fill(0); }
function zeros2D(r, c) { return Array.from({length: r}, () => new Array(c).fill(0)); }

// After
function zeros1D(n) { return new Float32Array(n); }  // Float32Array is zero-initialized
function zeros2D(r, c) { return Array.from({length: r}, () => new Float32Array(c)); }
```

### `createZeroGrads(params, config)`
- All gradient arrays become Float32Array
- Structure mirrors params but with zeros

### Backward functions
- `softmaxBackward`, `reluBackward`, `layerNormBackward`, `matVecBackward` — output Float32Array
- `addMat` helper — works with Array of Float32Array rows

### New activation backward functions
- `siluBackward(x, dOut)` — for SiLU activation
- `geluBackward(x, dOut)` — for GELU activation
- Selected by `config.activation` (default: `"silu"`)

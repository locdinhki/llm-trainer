# 1B: Transformer Initialization — Float32Array

## File
`src/model/transformer.js`

## Changes

### `createTinyTransformer(config)`
- All weight matrices initialized as Float32Array
- 2D matrices: Array of Float32Array rows
- Pattern: `Array.from({length: rows}, () => Float32Array.from({length: cols}, randn))`

### `forward(params, tokens, config)`
- Update to work with typed array weights
- Intermediate arrays (hidden states, Q/K/V, attention outputs) become Float32Array
- `tokens.map(...)` stays as regular Array (token indices are integers)

### `countParameters(params)`
- Update `Array.isArray()` check — Float32Array is not detected by `Array.isArray()`
- Use `arr.length !== undefined` or check `arr[0]` type instead

## Example
```js
// Before
embedding: Array.from({ length: vocabSize }, () =>
  Array.from({ length: embedDim }, randn)
)

// After
embedding: Array.from({ length: vocabSize }, () =>
  Float32Array.from({ length: embedDim }, randn)
)
```

# 1D: Forward Cache — Float32Array

## File
`src/model/forwardWithCache.js`

## Changes
- All intermediate arrays cached for backward pass become Float32Array
- Cache structure unchanged, just typed arrays inside:

```js
cache = {
  tokens,          // regular Array of ints
  blocks: [{
    preNorm1,      // Array of Float32Array
    normed1,       // Array of Float32Array
    Q, K, V,       // Array of Float32Array
    attnOutputs,   // Array of Float32Array
    attnWeights,   // Array of regular Array (softmax output)
    preNorm2,      // Array of Float32Array
    normed2,       // Array of Float32Array
    ffn1Pre,       // Array of Float32Array
    ffn1,          // Array of Float32Array
  }],
  lastHidden,      // Float32Array
  logits,          // Float32Array
}
```

## Notes
- Attention weights can stay as regular Arrays (they're small and used for visualization)
- The main performance gain is in the hidden state arrays that flow through matVec operations

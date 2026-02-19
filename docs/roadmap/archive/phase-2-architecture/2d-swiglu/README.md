# 2D: SwiGLU FFN (Gated Feed-Forward Network)

## What It Replaces
Simple two-layer FFN:
```
FFN(x) = W2(silu(W1(x) + b1)) + b2
```

## How SwiGLU Works
Adds a gating mechanism with a third weight matrix:
```
gate = silu(W_gate @ x + b_gate)
up   = W_up @ x + b_up
FFN(x) = W_down @ (gate * up) + b_down
```

The element-wise multiply `gate * up` allows the network to learn which features to pass through, producing better representations.

## Why It's Better
- **Gating**: The model learns what information to keep vs suppress
- **Better training dynamics**: Smoother loss landscape
- **Industry standard**: Used by LLaMA, LLaMA 2/3, Mistral, Gemma, PaLM
- **Trade-off**: One extra weight matrix per block (W_gate), ~33% more FFN params

## Parameter Changes Per Block

| Parameter | Current (Simple FFN) | SwiGLU FFN |
|-----------|---------------------|------------|
| W1 / W_up | [ffnDim, embedDim] | [ffnDim, embedDim] |
| b1 / b_up | [ffnDim] | [ffnDim] |
| W_gate | - | [ffnDim, embedDim] (NEW) |
| b_gate | - | [ffnDim] (NEW) |
| W2 / W_down | [embedDim, ffnDim] | [embedDim, ffnDim] |
| b2 / b_down | [embedDim] | [embedDim] |

## Implementation

### `src/model/transformer.js` — `createTinyTransformer()`
Add to each block:
```js
W_gate: Array.from({ length: ffnDim }, () => Float32Array.from({ length: embedDim }, randn)),
b_gate: new Float32Array(ffnDim),
```

Rename `W1`/`b1` to `W_up`/`b_up` for clarity (or keep W1 and add W_gate).

### Forward pass
```js
// Before (simple FFN)
const ffn1 = silu(vecAdd(matVec(block.W1, normed2), block.b1));
const ffn2 = vecAdd(matVec(block.W2, ffn1), block.b2);

// After (SwiGLU)
const gate = silu(vecAdd(matVec(block.W_gate, normed2), block.b_gate));
const up   = vecAdd(matVec(block.W1, normed2), block.b1);
const gated = vecMul(gate, up);  // element-wise multiply
const ffn_out = vecAdd(matVec(block.W2, gated), block.b2);
```

### Backward pass
The backward pass needs the **product rule** for the element-wise multiply:
```
dGate = dGated * up
dUp   = dGated * gate
```
Then `dGate` flows back through `siluBackward` and `matVecBackward(W_gate, ...)`, and `dUp` flows back through `matVecBackward(W1, ...)`.

### New math function needed
`vecMul(a, b)`: element-wise multiply returning Float32Array.

# 2C: SiLU/GELU Activation Functions

## What It Replaces
ReLU: `relu(x) = max(0, x)` — simple but has "dead neuron" problem (gradient is exactly 0 for x < 0).

## SiLU (Sigmoid Linear Unit / Swish)
Default activation for LLaMA-style models.

```
silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

**Backward:**
```
sigmoid = 1 / (1 + exp(-x))
siluBackward(x, dOut) = dOut * sigmoid * (1 + x * (1 - sigmoid))
```

**Properties:**
- Smooth, non-monotonic (dips slightly below 0 for negative x)
- No dead neurons — always has non-zero gradient
- Used by: LLaMA, LLaMA 2/3, Mistral, Gemma

## GELU (Gaussian Error Linear Unit)
Used by GPT-2, BERT, GPT-3.

```
gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

**Backward:**
```
Uses the chain rule through tanh — more complex but well-defined.
```

**Properties:**
- Smooth approximation of ReLU
- Slightly different shape from SiLU
- Used by: GPT-2, BERT, GPT-3, RoBERTa

## Implementation

### `src/model/math.js`
Add `silu(x)` and `gelu(x)` as element-wise Float32Array functions alongside existing `relu(x)`.

### `src/model/backward.js`
Add `siluBackward(x, dOut)` and `geluBackward(x, dOut)`.

### Config
`config.activation = "silu" | "gelu" | "relu"` (default: `"silu"`)

Forward and backward passes select the activation function based on config.

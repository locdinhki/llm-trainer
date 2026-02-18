# Phase 2: Architecture Modernization

## Goal
Bring the transformer architecture in line with modern standards (LLaMA/Mistral-style). This is the most educationally impactful phase — students see how real LLMs differ from the textbook transformer.

## What Changes

| Component | Current (Textbook) | Modern (LLaMA-style) | Used By |
|-----------|-------------------|---------------------|---------|
| Position encoding | Absolute learned embeddings | RoPE (rotary) | LLaMA, Mistral, Qwen |
| Normalization | LayerNorm (mean + variance) | RMSNorm (RMS only) | LLaMA, Mistral, Gemma |
| Activation | ReLU | SiLU/Swish (configurable) | LLaMA, Mistral |
| FFN structure | W2(act(W1(x))) | SwiGLU: W2(silu(W_gate(x)) * W_up(x)) | LLaMA, Mistral, Gemma |
| KV cache | None (full recompute) | Cache K/V for O(1) generation | All production models |
| Norm placement | Pre-norm (already correct) | Pre-norm (no change needed) | LLaMA, GPT-2+ |

## Subphases

| # | Subphase | Description |
|---|----------|-------------|
| 2A | [RoPE](2a-rope/) | Rotary Position Embeddings — replace absolute pos embeddings |
| 2B | [RMSNorm](2b-rmsnorm/) | Root Mean Square Normalization — replace LayerNorm |
| 2C | [SiLU/GELU](2c-silu-gelu/) | Modern activation functions — replace ReLU |
| 2D | [SwiGLU](2d-swiglu/) | Gated FFN — replace simple two-layer FFN |
| 2E | [KV Cache](2e-kv-cache/) | Key-Value cache for efficient autoregressive generation |

## Files Modified
- `src/model/rope.js` (new)
- `src/model/generation.js` (new)
- `src/model/math.js`
- `src/model/transformer.js`
- `src/model/forwardWithCache.js`
- `src/model/backward.js`
- `src/model/training.js`

## Verification
- [ ] Gradient check passes after ALL architecture changes (relative error < 1e-4)
- [ ] Model has no `posEmbedding` param (RoPE replaces it)
- [ ] Blocks have `ln1_g`/`ln2_g` but no `ln1_b`/`ln2_b` (RMSNorm)
- [ ] Default activation is SiLU, configurable to GELU/ReLU
- [ ] Each block has `W_gate` alongside `W1`/`W2` (SwiGLU)
- [ ] Training still converges on default sentences

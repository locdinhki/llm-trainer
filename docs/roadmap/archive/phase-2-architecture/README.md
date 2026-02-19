# Phase 2: Architecture Modernization — Done

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

| # | Subphase | Description | Status |
|---|----------|-------------|--------|
| 2A | [RoPE](2a-rope/) | Rotary Position Embeddings — replace absolute pos embeddings | Done |
| 2B | [RMSNorm](2b-rmsnorm/) | Root Mean Square Normalization — replace LayerNorm | Done |
| 2C | [SiLU/GELU](2c-silu-gelu/) | Modern activation functions — replace ReLU | Done |
| 2D | [SwiGLU](2d-swiglu/) | Gated FFN — replace simple two-layer FFN | Done |
| 2E | [KV Cache](2e-kv-cache/) | Key-Value cache for efficient autoregressive generation | Done |

## Files Modified
- `src/model/rope.js` (new)
- `src/model/generation.js` (new)
- `src/model/math.js`
- `src/model/transformer.js`
- `src/model/forwardWithCache.js`
- `src/model/backward.js`
- `src/model/training.js`
- `src/model/training.worker.js`

## Key Technical Notes
- Implementation order was 2C → 2B → 2A → 2D → 2E (simplest first for incremental verification)
- SwiGLU (2D) hardcodes SiLU for the gate activation, making the configurable activation setting unnecessary — it was removed
- RoPE backward is the inverse rotation (transpose of rotation matrix), placed between attention gradient computation and Q/K projection backward
- Gradient verification eps increased from 1e-4 to 1e-3 and threshold from 5e-2 to 1e-1 for Float32Array precision
- KV cache is inference-only (not used during training); will be wired into UI in Phase 7
- `vecMul` (element-wise multiply) added to math.js for SwiGLU product rule

## Verification
- [x] `npm run build` compiles cleanly
- [x] Gradient check passes (1 block: max relErr 1.7e-2, 2 blocks: max relErr 5.1e-2)
- [x] Model has no `posEmbedding` param (RoPE replaces it)
- [x] Blocks have `ln1_g`/`ln2_g` but no `ln1_b`/`ln2_b` (RMSNorm)
- [x] SwiGLU uses SiLU gate activation by default
- [x] Each block has `W_gate` alongside `W1`/`W2` (SwiGLU)
- [x] Training converges on default sentences (loss 2.25 → 0.45 in 200 steps)
- [x] KV cache produces identical logits to full forward pass (max diff ~1.2e-7)

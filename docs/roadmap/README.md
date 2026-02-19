# Scale-Up Roadmap

Transform the tiny educational transformer (~3K params, 11 words) into a near-production architecture (~87K params, 200 BPE tokens) while keeping everything in-browser and from-scratch JavaScript.

## Target Specifications

| Aspect | Current | After Scale-Up |
|--------|---------|---------------|
| Tokenization | Word split (~11 tokens) | BPE (200 tokens) |
| Embed dim | 16 | 48 |
| Attention heads | 2 | 4 (head dim=12) |
| FFN dim | 32 | 128 |
| Blocks | 1 | 3 |
| Seq length | 10 | 32 |
| Parameters | ~2,700 | ~87,000 |
| Training sentences | 6 | ~150 |
| Batch size | all 28 pairs | mini-batch 32 |
| Time per step | <1ms | ~30-80ms |
| Steps to converge | ~200 | ~2,000-5,000 |
| Optimizer | SGD | AdamW |
| Activation | ReLU | SiLU (configurable) |
| Normalization | LayerNorm | RMSNorm |
| Position encoding | Absolute learned | RoPE (rotary) |
| KV cache | None | Yes (for generation) |
| Array type | JS Array | Float32Array |

## Phases

| # | Phase | Description | Status |
|---|-------|-------------|--------|
| 1 | [Float32Array Migration](archive/phase-1-float32array/) | Replace JS Arrays with typed arrays for 2-4x speedup | Done |
| 2 | [Architecture Modernization](archive/phase-2-architecture/) | RoPE, RMSNorm, SiLU, SwiGLU, KV cache | Done |
| 3 | [Mini-Batch + Adam](archive/phase-3-optimizer/) | Mini-batch SGD, Adam optimizer, LR schedule | Done |
| 4 | [Training Observability](archive/phase-4-observability/) | AdamW, gradient norms, perplexity, LR curves, checkpoints | Done |
| 5 | [BPE Tokenizer](archive/phase-5-bpe/) | Byte-Pair Encoding from scratch | Done |
| 6 | [Expanded Corpus](archive/phase-6-corpus/) | ~150 curated sentences with preset selector | Done |
| 7 | [Model Scaling + Worker](archive/phase-7-scaling/) | Expanded config, Web Worker integration | Done |
| 8 | [Interactive Generation](archive/phase-8-generation/) | Text generation panel with temperature/top-k/top-p | Done |
| 9 | [Visualization Adaptations](phase-9-visualization/) | Adapt all panels for larger vocab/sequences | Pending |

## Implementation Order

```
Phase 1 (Float32Array)          <- Foundation, do first
  |
Phase 2 (Architecture: RoPE, RMSNorm, SiLU, SwiGLU, KV cache)
  |
Phase 3 (Mini-batch + Adam)     <- Needs typed arrays + new architecture
  |
Phase 4 (Observability)         <- AdamW, metrics, checkpoints
  |
Phase 5 (BPE)                   <- Needs mini-batch (too much data for full-batch)
  |
Phase 6 (Corpus)                <- Needs BPE tokenizer
  |
Phase 7 (Scaling + Worker)      <- Needs everything above
  |
Phase 8 (Generation + KV cache) <- Needs trained model + tokenizer
  |
Phase 9 (Viz adaptations)       <- Polish, do last
```

Build and verify after each phase. Each phase should leave the app fully functional.

## Verification Checklist

After each phase:
- [ ] `npm run build` compiles cleanly
- [ ] App loads, trains, and visualizes without errors
- [ ] Gradient verification passes (relative error < 5e-2 for Float32)

End-to-end (after all phases):
- [ ] "Stories" corpus + BPE tokenization trains on 150 sentences
- [ ] "Medium (90K)" preset shows ~87K params
- [ ] Training runs at 10-30 steps/second on M4 Mac
- [ ] Loss decreases steadily, perplexity drops
- [ ] Generation panel produces coherent sentence fragments
- [ ] KV cache generation produces tokens one at a time
- [ ] All visualizations work at 32-token seqLen
- [ ] "Simple (6 sentences)" + Word-level backward compatible
- [ ] Web Worker: training off main thread, UI smooth at 60fps

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
| Optimizer | SGD | Adam |
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
| 4 | [BPE Tokenizer](phase-4-bpe/) | Byte-Pair Encoding from scratch | Pending |
| 5 | [Expanded Corpus](phase-5-corpus/) | ~150 curated sentences with preset selector | Pending |
| 6 | [Model Scaling + Worker](phase-6-scaling/) | Expanded config, dropout, Web Worker integration | Pending |
| 7 | [Interactive Generation](phase-7-generation/) | Text generation panel with temperature/top-k | Pending |
| 8 | [Visualization Adaptations](phase-8-visualization/) | Adapt all panels for larger vocab/sequences | Pending |

## Implementation Order

```
Phase 1 (Float32Array)          <- Foundation, do first
  |
Phase 2 (Architecture: RoPE, RMSNorm, SiLU, SwiGLU, KV cache)
  |
Phase 3 (Mini-batch + Adam)     <- Needs typed arrays + new architecture
  |
Phase 4 (BPE)                   <- Needs mini-batch (too much data for full-batch)
  |
Phase 5 (Corpus)                <- Needs BPE tokenizer
  |
Phase 6 (Scaling + Worker)      <- Needs everything above
  |
Phase 7 (Generation + KV cache) <- Needs trained model + tokenizer
  |
Phase 8 (Viz adaptations)       <- Polish, do last
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

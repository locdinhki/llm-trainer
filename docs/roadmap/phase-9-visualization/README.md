# Phase 9: Visualization Adaptations

## Goal
Adapt all visualization panels to work well with larger vocabularies (200 BPE tokens), longer sequences (32 tokens), and more attention heads/blocks (4 heads x 3 blocks).

## Subphases

| # | Subphase | Component | Key Change |
|---|----------|-----------|------------|
| 9A | [Prediction Panel](9a-prediction-panel/) | PredictionPanel.jsx | Top-K view instead of showing all tokens |
| 9B | [Embedding Panel](9b-embedding-panel/) | EmbeddingPanel.jsx | Hover labels, category filter |
| 9C | [Attention Panel](9c-attention-panel/) | AttentionPanel.jsx | Canvas renderer, overview grid |
| 9D | [Loss Panel](9d-loss-panel/) | LossPanel.jsx | Perplexity, smoothed loss line |

## Files
- `src/components/PredictionPanel.jsx`
- `src/components/EmbeddingPanel.jsx`
- `src/components/AttentionPanel.jsx`
- `src/components/LossPanel.jsx`

## Verification
- [ ] All panels render correctly at vocabSize=200, seqLen=32
- [ ] No performance issues (DOM element count stays reasonable)
- [ ] BPE subword tokens displayed with correct styling
- [ ] Still works at small scale (vocabSize=11, seqLen=10)

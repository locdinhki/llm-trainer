# Phase 8: Interactive Generation Panel

## Goal
Add a text generation panel where users type a prompt and watch the model generate text autoregressively — token by token, just like ChatGPT. This is the "wow moment" of the dashboard.

## How Autoregressive Generation Works
1. User types initial tokens: "the cat"
2. Model predicts next token probabilities
3. Sample from probabilities (with temperature): "sat"
4. Append "sat" to sequence: "the cat sat"
5. Repeat until max length or stopping condition

## Subphases

| # | Subphase | Description |
|---|----------|-------------|
| 8A | [Generation Panel](8a-generation-panel/) | React component with input, output, and controls |
| 8B | [Sampling Utilities](8b-sampling/) | Token sampling with temperature and top-k |

## Files
- `src/components/GenerationPanel.jsx` (new)
- `src/model/generation.js` (new — also contains KV cache from Phase 2E)
- `src/App.jsx`

## Verification
- [ ] Type "the" -> model generates a coherent continuation
- [ ] Temperature slider affects output diversity (low = deterministic, high = random)
- [ ] Top-k sampling limits to top candidates
- [ ] Top-p (nucleus) sampling limits to cumulative probability threshold
- [ ] Generated tokens appear one at a time with probability annotations
- [ ] KV cache makes generation visibly fast
- [ ] Unknown word input shows error (word-level) or handles gracefully (BPE)

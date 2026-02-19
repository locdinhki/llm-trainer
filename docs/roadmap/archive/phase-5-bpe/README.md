# Phase 5: BPE Tokenizer

## Goal
Implement Byte-Pair Encoding tokenization from scratch. BPE is the tokenization method used by GPT-2, GPT-3, GPT-4, LLaMA, and most modern LLMs.

## What Is BPE?
Instead of treating whole words as tokens, BPE builds a vocabulary of subword units by iteratively merging the most frequent adjacent pairs:

1. Start with individual characters as tokens: `t`, `h`, `e`, ` `, `c`, `a`, `t`
2. Count all adjacent pairs, find most frequent (e.g., `t`+`h` appears 50 times)
3. Merge into new token `th`, update all sequences
4. Repeat until desired vocabulary size reached

Result: a mix of characters, common subwords, and frequent whole words.

## Why BPE?
- **Open vocabulary**: Can represent ANY text, not just known words
- **Efficient**: Common words are single tokens, rare words are split into pieces
- **Industry standard**: Used by virtually every modern LLM
- **Educational**: Understanding BPE is essential for understanding modern AI

## Subphases

| # | Subphase | Description |
|---|----------|-------------|
| 5A | [BPE Algorithm](5a-bpe-algorithm/) | Core BPE training, encoding, and decoding |
| 5B | [Data Integration](5b-data-integration/) | Unified tokenizer abstraction for word-level and BPE |
| 5C | [Settings UI](5c-settings-ui/) | Toggle and config in SettingsPanel |

## Files
- `src/model/bpe.js` (new)
- `src/model/data.js`
- `src/components/SettingsPanel.jsx`

## Verification
- [ ] BPE trains on corpus in < 100ms
- [ ] Encoding/decoding round-trips correctly: `decode(encode(text)) === text`
- [ ] Training data correctly generated from BPE-tokenized sentences
- [ ] Model trains and converges with BPE tokens
- [ ] Word-level mode still works (backward compatible)

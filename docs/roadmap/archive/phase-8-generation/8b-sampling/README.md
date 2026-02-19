# 8B: Sampling Utilities

## File: `src/model/generation.js`

This file also contains the KV cache code from Phase 2E.

## Functions

### `sampleToken(probs, temperature, topK, topP)`
```js
1. Apply temperature: logits[i] /= temperature (before softmax)
   - Or equivalently: probs[i]^(1/temperature), then renormalize
2. Sort by probability, keep only top-k
3. Apply top-p (nucleus): sort by probability descending, accumulate until sum >= p, discard rest
4. Renormalize remaining probabilities
5. Sample from the distribution:
   - Generate random number r in [0, 1)
   - Walk through cumulative probabilities until cumSum > r
6. Return { tokenId, probability }
```

### `generate(params, config, tokenizer, inputText, maxTokens, temperature, topK, topP)`
```js
1. Encode input text: tokens = tokenizer.encode(inputText)
2. Create KV cache: cache = createKVCache(config)
3. Prefill: run forward for all input tokens, populate cache
4. For i = 0 to maxTokens:
   a. Forward pass for last token (using KV cache)
   b. Get logits, apply softmax
   c. Sample next token: sampleToken(probs, temperature, topK, topP)
   d. Append token to sequence
   e. Yield { token, probability, sequence } (for streaming display)
   f. Stop if maxProb < 0.05 (model is too uncertain)
5. Return full generated sequence
```

### Temperature Intuition
| Temperature | Behavior |
|------------|----------|
| 0.1 | Nearly greedy — always picks the most likely token |
| 0.5 | Somewhat diverse — occasionally picks 2nd/3rd choice |
| 1.0 | Standard sampling — matches training distribution |
| 1.5 | Creative — more random, sometimes surprising |
| 2.0 | Very random — may produce incoherent output |

### Top-k Intuition
| Top-k | Behavior |
|-------|----------|
| 1 | Greedy decoding (always picks #1) |
| 5 | Choose from top 5 candidates |
| 10 | More variety |
| vocabSize | No filtering (pure temperature sampling) |

### Top-P Intuition
| Top-P | Behavior |
|-------|----------|
| 0.1 | Very conservative (only top ~1-2 tokens) |
| 0.5 | Moderate filtering |
| 0.9 | Standard nucleus sampling |
| 1.0 | No filtering (pure temperature + top-k) |

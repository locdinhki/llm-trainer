# 8A: Generation Panel Component

## New File: `src/components/GenerationPanel.jsx`

## UI Layout
```
+-------------------------------------------+
| Text Generation                           |
+-------------------------------------------+
| [Type words to start...        ] [Generate]|
|                                            |
| the cat sat on the mat                     |
|  100%  92%  87%  95%  91%  88%             |
|                                            |
| Temperature: [====O=======] 1.0            |
| Top-k:       [==O=========] 5             |
| Top-p:       [========O===] 0.9            |
|                                 [Clear]    |
+-------------------------------------------+
```

## Features

### Text Input
- Input field for user prompt
- Accepts space-separated words (word-level mode) or free text (BPE mode)
- Validates input: unknown words show inline error

### Generate Button
- Triggers autoregressive generation loop
- Generates up to `seqLen - inputLen` additional tokens
- Each token appears with a short delay (~100ms) for visual effect
- Stops early if model is highly uncertain (max prob < 5%)

### Generated Output
- Shows full generated sequence
- Each token annotated with its probability (small text below)
- Input tokens shown in white, generated tokens in a highlight color
- Probability < 50% shown in orange, < 20% in red

### Controls
- **Temperature slider** (0.1 to 2.0, default 1.0):
  - Low (0.1): Nearly deterministic, always picks highest-prob token
  - Medium (1.0): Balanced randomness
  - High (2.0): Very random, creative but potentially incoherent
- **Top-k slider** (1 to vocab_size, default 5):
  - Only sample from the top-k most probable tokens
  - k=1 is greedy decoding
- **Top-p slider** (0.1 to 1.0, default 0.9):
  - Only sample from the smallest set of tokens whose cumulative probability exceeds p
  - p=0.1: Very focused
  - p=0.9: Default nucleus sampling
  - p=1.0: No filtering
- **Clear button**: Reset output

### Integration with App.jsx
- Panel receives `params`, `config`, `tokenizer` from App
- Uses `forwardWithKVCache` from `generation.js` for efficient generation
- Generation runs on main thread (fast enough at this scale)
- Does NOT interfere with training — can generate while training is paused

# 4C: Settings UI — Tokenization Controls

## File
`src/components/SettingsPanel.jsx`

## New Controls

### Tokenization Mode Toggle
- Segmented control: **"Word-level"** / **"BPE"**
- Default: "Word-level" (backward compatible)
- Changing mode triggers full reset (new vocab, new model, new training data)

### BPE Vocabulary Size Slider
- Only visible when BPE mode is selected
- Range: 50 to 500
- Default: 200
- Shows current value
- Changing triggers BPE retrain + model reset

### Tokenizer Stats Display
- Show below the tokenization controls:
  - Vocabulary size: e.g., "200 tokens"
  - Merge count: e.g., "160 merges" (BPE only)
  - Avg tokens per sentence: e.g., "8.3 tokens/sentence"
  - Example tokenization: show one sentence tokenized with color-coded tokens

# 5B: Corpus Selector UI

## File
`src/components/SettingsPanel.jsx`

## Implementation

### Corpus Dropdown
- Options: "Simple (6 sentences)" / "Stories (~150 sentences)" / "Custom"
- Default: "Simple"
- Changing preset loads the corresponding sentence array
- "Custom" enables the existing textarea for user-authored text

### Corpus Statistics
Display below the dropdown:
- Sentence count: e.g., "150 sentences"
- Total words: e.g., "2,341 words"
- Unique words: e.g., "187 unique words"
- Avg sentence length: e.g., "15.6 words/sentence"

### Behavior on Change
- Changing corpus triggers full reset:
  1. Update sentences state
  2. Rebuild vocabulary/tokenizer
  3. Reinitialize model (new vocabSize may differ)
  4. Reset training step to 0
  5. Clear loss history
  6. Log: `"[Config] Loaded 'Stories' corpus (150 sentences, 187 unique words)"`

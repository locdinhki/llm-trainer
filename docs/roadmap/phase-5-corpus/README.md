# Phase 5: Expanded Corpus

## Goal
Provide ~150 curated training sentences as preset corpora, giving the model enough data to learn interesting patterns while keeping training interactive.

## Why Expand?
- Current 6 sentences only teach ~11 words — very limited generation capability
- 150 sentences with ~200 unique words enables basic grammatical patterns (subject-verb-object, articles before nouns) and topical coherence
- BPE tokenizer (Phase 4) needs more diverse text to produce meaningful subword merges

## Subphases

| # | Subphase | Description |
|---|----------|-------------|
| 5A | [Preset Corpora](5a-preset-corpora/) | Curated sentence collections |
| 5B | [Corpus Selector](5b-corpus-selector/) | UI dropdown + stats display |

## Files
- `src/model/data.js`
- `src/components/SettingsPanel.jsx`

## Verification
- [ ] "Simple" preset loads original 6 sentences (backward compatible)
- [ ] "Stories" preset loads ~150 sentences
- [ ] Custom text area still works
- [ ] Corpus stats display correctly
- [ ] Model trains and converges on expanded corpus

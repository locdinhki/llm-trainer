# 4B: Data Integration — Unified Tokenizer Abstraction

## File
`src/model/data.js`

## Tokenizer Interface
Both word-level and BPE tokenizers implement the same interface:

```js
{
  encode(text) -> number[],       // text -> token IDs
  decode(ids) -> string,          // token IDs -> text
  vocabSize: number,              // total vocabulary size
  id2token: string[],             // index -> token string
  token2id: { [token]: number },  // token string -> index
}
```

## New Functions

### `buildBPETokenizer(sentences, targetVocabSize)`
- Joins all sentences into corpus text
- Calls `trainBPE()` from `bpe.js`
- Wraps result in tokenizer interface
- Returns tokenizer object

### `buildWordTokenizer(sentences)`
- Wraps current `buildVocabulary()` logic in tokenizer interface
- `encode(text)` = `text.split(" ").map(w => token2id[w])`
- `decode(ids)` = `ids.map(id => id2token[id]).join(" ")`
- Backward compatible with existing behavior

## Updated Functions

### `makeTrainingData(sentences, tokenizer)`
- Uses `tokenizer.encode(sentence)` instead of `sentence.split(" ").map(w => word2id[w])`
- Training pairs: all prefix -> next-token pairs from encoded sequences

### `generatePrompts(sentences, tokenizer)`
- Encodes each sentence, creates prompts from prefixes
- Labels show decoded tokens (may be subword pieces)
- Target is the decoded last token

## Token Categories for BPE
- Full-word tokens: look up in `KNOWN_CATEGORIES` (animal, object, verb, grammar)
- Subword tokens (partial words): category = `"subword"`
- New color: `CATEGORY_COLORS.subword = "#c084fc"` (purple)

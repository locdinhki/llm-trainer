# 4A: BPE Algorithm

## New File: `src/model/bpe.js` (~150 lines)

Implemented entirely from scratch, no dependencies.

## Core Functions

### `trainBPE(text, targetVocabSize)`
```js
1. Initialize base vocabulary: individual characters (a-z, A-Z, 0-9, space, punctuation)
2. Split entire text into character-level token sequences (one per word/sentence)
3. While vocab.size < targetVocabSize:
   a. Count all adjacent token pairs across all sequences
   b. Find the pair with highest frequency
   c. Create new token = pair[0] + pair[1]
   d. Add to vocabulary
   e. Replace all occurrences of the pair in all sequences
   f. Record merge rule: (pair[0], pair[1]) -> newToken
4. Return { merges: [...], vocab: Map, id2token: [...], token2id: Map }
```

### `encode(text, merges, token2id)`
```js
1. Split text into characters
2. Apply merge rules in order (greedy left-to-right):
   For each merge (a, b) -> ab:
     Scan sequence, replace adjacent (a, b) with ab
3. Map tokens to IDs via token2id
4. Return array of token IDs
```

### `decode(ids, id2token)`
```js
1. Map IDs to token strings
2. Join with empty string (BPE tokens include spaces as part of the token)
3. Return decoded string
```

## Example
With text "the cat sat on the mat":

After character initialization: `['t','h','e',' ','c','a','t',' ','s','a','t',...]`

Merges might be:
1. `('t','h')` -> `'th'` (most common pair)
2. `('th','e')` -> `'the'`
3. `(' ','the')` -> `' the'`
4. `('a','t')` -> `'at'`
5. `('s','at')` -> `'sat'`
...

Final vocab (~200 tokens): mix of characters, subwords like `'th'`, `'at'`, and full words like `'the'`, `'cat'`.

## Performance
For ~2,500 words of text with target vocab 200:
- ~160 merge iterations
- Each iteration scans all token pairs
- Total: < 100ms in the browser

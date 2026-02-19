/**
 * Byte-Pair Encoding (BPE) tokenizer — implemented from scratch.
 *
 * The same algorithm used by GPT-2/3/4, LLaMA, and most modern LLMs.
 * Spaces are kept as part of tokens (prepended to non-first words) so that
 * decode(encode(text)) === text always holds.
 */

// ── Training ────────────────────────────────────────────────────────────────

/**
 * Train a BPE vocabulary on the given text.
 *
 * @param {string}  text             The full corpus text
 * @param {number}  targetVocabSize  Desired vocabulary size (e.g. 200)
 * @returns {{ merges: [string,string][], id2token: string[], token2id: Object }}
 */
export function trainBPE(text, targetVocabSize) {
  // 1. Split corpus into "words" (whitespace-delimited) and prepend a space
  //    to every word except the very first one.  This keeps spaces as part of
  //    tokens so round-tripping works: decode(encode(t)) === t.
  const rawWords = text.split(/\s+/).filter(Boolean);
  // wordSequences[i] = array of character-level tokens for one "word"
  const wordFreqs = new Map();          // word-string -> frequency
  for (let i = 0; i < rawWords.length; i++) {
    const w = (i === 0 ? "" : " ") + rawWords[i];
    wordFreqs.set(w, (wordFreqs.get(w) || 0) + 1);
  }

  // Represent each unique word as an array of single-char tokens, paired with freq
  let words = [];   // { chars: string[], freq: number }[]
  for (const [w, freq] of wordFreqs) {
    words.push({ chars: [...w], freq });
  }

  // 2. Build base vocabulary from all unique characters
  const vocab = new Set();
  for (const { chars } of words) {
    for (const ch of chars) vocab.add(ch);
  }

  const merges = [];  // ordered list of merge rules: [tokenA, tokenB]

  // 3. Iteratively merge the most frequent adjacent pair
  while (vocab.size < targetVocabSize) {
    // Count adjacent pairs weighted by word frequency
    const pairCounts = new Map();
    for (const { chars, freq } of words) {
      for (let j = 0; j < chars.length - 1; j++) {
        const key = chars[j] + "\0" + chars[j + 1];
        pairCounts.set(key, (pairCounts.get(key) || 0) + freq);
      }
    }

    if (pairCounts.size === 0) break;  // nothing left to merge

    // Find the pair with the highest count
    let bestKey = null;
    let bestCount = -1;
    for (const [key, count] of pairCounts) {
      if (count > bestCount) { bestCount = count; bestKey = key; }
    }

    const [a, b] = bestKey.split("\0");
    const merged = a + b;

    // Record merge rule & add new token to vocab
    merges.push([a, b]);
    vocab.add(merged);

    // Apply merge to every word sequence
    for (const word of words) {
      const c = word.chars;
      let i = 0;
      while (i < c.length - 1) {
        if (c[i] === a && c[i + 1] === b) {
          c.splice(i, 2, merged);
          // don't advance i — the new token might pair with the next one
        } else {
          i++;
        }
      }
    }
  }

  // 4. Build final token <-> id mappings
  //    Order: base characters first (sorted), then merged tokens in merge order
  const baseChars = [];
  const mergedTokens = new Set();
  for (const [a, b] of merges) mergedTokens.add(a + b);
  for (const t of vocab) {
    if (!mergedTokens.has(t)) baseChars.push(t);
  }
  baseChars.sort();

  const id2token = [...baseChars];
  for (const [a, b] of merges) id2token.push(a + b);

  const token2id = {};
  for (let i = 0; i < id2token.length; i++) token2id[id2token[i]] = i;

  return { merges, id2token, token2id };
}

// ── Encoding ────────────────────────────────────────────────────────────────

/**
 * Encode text into token IDs using a trained BPE vocabulary.
 *
 * @param {string}            text      Input text
 * @param {[string,string][]} merges    Ordered merge rules from trainBPE
 * @param {Object}            token2id  Token-string -> ID mapping
 * @returns {number[]}  Array of token IDs
 */
export function encode(text, merges, token2id) {
  // Split into "words" with leading-space convention (same as training)
  const rawWords = text.split(/\s+/).filter(Boolean);
  const ids = [];

  for (let wi = 0; wi < rawWords.length; wi++) {
    let chars = [...((wi === 0 ? "" : " ") + rawWords[wi])];

    // Apply merge rules in order
    for (const [a, b] of merges) {
      let i = 0;
      while (i < chars.length - 1) {
        if (chars[i] === a && chars[i + 1] === b) {
          chars.splice(i, 2, a + b);
        } else {
          i++;
        }
      }
    }

    // Map tokens to IDs
    for (const tok of chars) {
      if (tok in token2id) {
        ids.push(token2id[tok]);
      }
      // Unknown characters are silently dropped (shouldn't happen if trained on same corpus)
    }
  }

  return ids;
}

// ── Decoding ────────────────────────────────────────────────────────────────

/**
 * Decode token IDs back to text.
 *
 * @param {number[]}  ids       Array of token IDs
 * @param {string[]}  id2token  ID -> token-string mapping
 * @returns {string}  Decoded text
 */
export function decode(ids, id2token) {
  return ids.map((id) => id2token[id]).join("");
}

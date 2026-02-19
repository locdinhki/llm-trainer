import { trainBPE, encode, decode } from "./bpe.js";

export const DEFAULT_SENTENCES = [
  "the cat sat on the mat",
  "the dog sat on the rug",
  "the cat ate the fish",
  "the dog ate the bone",
  "the cat is on the mat",
  "the dog is on the rug",
];

const KNOWN_CATEGORIES = {
  cat: "animal", dog: "animal",
  mat: "object", rug: "object", fish: "object", bone: "object",
  sat: "verb", ate: "verb", is: "verb",
  the: "grammar", on: "grammar", a: "grammar", an: "grammar",
  in: "grammar", to: "grammar", for: "grammar", with: "grammar",
  from: "grammar", by: "grammar", of: "grammar", up: "grammar",
  are: "grammar", was: "grammar", were: "grammar",
};

export const CATEGORY_COLORS = {
  animal: "#4ade80",
  object: "#fb923c",
  verb: "#60a5fa",
  grammar: "#94a3b8",
  other: "#e879f9",
  subword: "#c084fc",
};

// ── Unified tokenizer interface ─────────────────────────────────────────────
// Both builders return the same shape:
//   { encode, decode, vocabSize, id2token, token2id, tokenCategories }

/**
 * Word-level tokenizer — wraps the original buildVocabulary logic.
 */
export function buildWordTokenizer(sentences) {
  const allWords = [...new Set(sentences.flatMap((s) => s.split(" ")))];
  const token2id = {};
  allWords.forEach((w, i) => (token2id[w] = i));

  const tokenCategories = {};
  allWords.forEach((w) => {
    tokenCategories[w] = KNOWN_CATEGORIES[w] || "other";
  });

  return {
    encode: (text) => text.split(" ").map((w) => token2id[w]),
    decode: (ids) => ids.map((id) => allWords[id]).join(" "),
    vocabSize: allWords.length,
    id2token: allWords,
    token2id,
    tokenCategories,
    mergeCount: 0,
  };
}

/**
 * BPE tokenizer — trains BPE on the corpus, returns unified interface.
 */
export function buildBPETokenizer(sentences, targetVocabSize = 200) {
  const corpus = sentences.join(" ");
  const { merges, id2token, token2id } = trainBPE(corpus, targetVocabSize);

  // Categorize tokens: if a token (trimmed) matches a known word, use that category.
  // Otherwise it's a subword fragment.
  const tokenCategories = {};
  for (const tok of id2token) {
    const trimmed = tok.trim();
    if (trimmed in KNOWN_CATEGORIES) {
      tokenCategories[tok] = KNOWN_CATEGORIES[trimmed];
    } else {
      tokenCategories[tok] = trimmed.length === tok.length && tok.length > 1 ? "other" : "subword";
    }
  }

  return {
    encode: (text) => encode(text, merges, token2id),
    decode: (ids) => decode(ids, id2token),
    vocabSize: id2token.length,
    id2token,
    token2id,
    tokenCategories,
    mergeCount: merges.length,
  };
}

// ── Backward-compatible wrappers (used by App.jsx before full migration) ────

export function buildVocabulary(sentences) {
  const tok = buildWordTokenizer(sentences);
  return {
    allWords: tok.id2token,
    word2id: tok.token2id,
    id2word: tok.id2token,
    vocabSize: tok.vocabSize,
    wordCategories: tok.tokenCategories,
  };
}

// ── Training data & prompts (now tokenizer-aware) ───────────────────────────

export function makeTrainingData(sentences, tokenizer) {
  const data = [];
  // If tokenizer is an object with .encode, use it. Otherwise treat as word2id map (backward compat).
  const enc = typeof tokenizer === "object" && tokenizer.encode
    ? tokenizer.encode
    : (text) => text.split(" ").map((w) => tokenizer[w]);

  for (const sent of sentences) {
    const tokens = enc(sent);
    for (let i = 1; i < tokens.length; i++) {
      const inputSeq = tokens.slice(0, i);
      data.push({ input: inputSeq, target: tokens[i], sentence: sent });
    }
  }
  return data;
}

export function generatePrompts(sentences, tokenizer) {
  const prompts = [];
  const seen = new Set();
  const enc = typeof tokenizer === "object" && tokenizer.encode
    ? tokenizer.encode
    : (text) => text.split(" ").map((w) => tokenizer[w]);
  const dec = typeof tokenizer === "object" && tokenizer.decode
    ? tokenizer.decode
    : null;

  for (const sent of sentences) {
    const allIds = enc(sent);
    if (allIds.length < 2) continue;
    const inputIds = allIds.slice(0, -1);
    const targetId = allIds[allIds.length - 1];
    const key = inputIds.join(",");
    if (seen.has(key)) continue;
    seen.add(key);

    // Build a readable label
    let label;
    let target;
    if (dec) {
      label = dec(inputIds).trim() + " ___";
      target = dec([targetId]).trim();
    } else {
      // Legacy word2id path
      const words = sent.split(" ");
      label = words.slice(0, -1).join(" ") + " ___";
      target = words[words.length - 1];
    }

    prompts.push({ tokens: inputIds, label, target, targetId });
  }
  return prompts;
}

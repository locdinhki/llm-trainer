import { trainBPE, encode, decode } from "./bpe.js";

// ── Preset Corpora ──────────────────────────────────────────────────────────

export const SIMPLE_SENTENCES = [
  "the cat sat on the mat",
  "the dog sat on the rug",
  "the cat ate the fish",
  "the dog ate the bone",
  "the cat is on the mat",
  "the dog is on the rug",
];

export const DEFAULT_SENTENCES = SIMPLE_SENTENCES;

export const EXPANDED_STORIES = [
  // ── Animals & Actions ──────────────────────────────
  "the big cat chased the small mouse",
  "the brown dog ran to the park",
  "a bird sat on the tall tree",
  "the fish swam in the cold river",
  "the horse ran across the green field",
  "a small rabbit hid under the bush",
  "the old owl sat in the dark tree",
  "the fox ran through the forest",
  "a duck swam on the blue lake",
  "the dog and the cat sat on the rug",
  "the bird flew over the tall hill",
  "a frog jumped into the cold pond",
  "the big bear walked near the river",
  "the hen sat on her nest in the barn",
  "the deer ran into the deep forest",
  "a small fish hid under the stone",
  "the wolf ran across the dark field",
  "the sheep stood on the green hill",
  "the dog chased the cat up the tree",
  "a brown bird sang in the garden",
  "the horse stood by the old barn",
  "the duck and the hen went to the pond",
  "a red fox hid in the tall grass",
  "the cat found a mouse near the door",
  "the big dog jumped over the wall",

  // ── People & Daily Life ────────────────────────────
  "the boy went to the school",
  "the girl read a good book",
  "the man walked to the store",
  "the woman made a big cake",
  "the child played in the garden",
  "the farmer fed the brown hen",
  "the baker made fresh bread",
  "the teacher read to the class",
  "the boy and the girl ran to the park",
  "the man sat by the old door",
  "the woman went to the store",
  "the child found a red ball",
  "the girl gave the boy a book",
  "the farmer walked to the barn",
  "the baker sold bread at the store",
  "the boy put on his new hat",
  "the man and the woman sat on the bench",
  "the teacher gave the child a gold star",
  "the girl played with the red ball",
  "the boy ran home from the school",
  "the farmer had a big field of corn",
  "the woman read a long book by the fire",
  "the child sat on the bed and read",
  "the man fed the dog in the yard",
  "the baker put the cake on the shelf",

  // ── Nature & Weather ───────────────────────────────
  "the sun rose in the east",
  "the rain fell on the green trees",
  "a cold wind blew from the north",
  "the moon shone over the dark lake",
  "the snow fell on the cold ground",
  "a bright star shone in the night sky",
  "the river ran through the deep forest",
  "the wind blew the leaves off the tree",
  "a white cloud drifted across the blue sky",
  "the sun set in the red west",
  "the rain made the grass very green",
  "the snow covered the old hill",
  "a warm wind came from the south",
  "the flowers grew in the bright sun",
  "the lake was still and very blue",
  "the dark clouds came from the north",
  "the leaves fell from the tall tree",
  "the river was cold and very deep",
  "a long road went through the forest",
  "the sky turned red at the end of the day",
  "the bright moon rose over the hill",
  "the warm sun shone on the field",
  "a bird sang as the sun rose",
  "the frost covered the ground at night",
  "the stream ran down the green hill",

  // ── Simple Narratives ──────────────────────────────
  "once upon a time there was a king",
  "the king had a big castle",
  "the queen sat in the garden",
  "the king and the queen had a feast",
  "a brave boy went on a long quest",
  "the boy found a gold key by the river",
  "he took the key to the old door",
  "the door led to a room full of gold",
  "the queen gave the boy a red cape",
  "the king sat on his throne of gold",
  "once there was a girl in a small town",
  "the girl found a bright stone by the road",
  "she took the stone to the wise old man",
  "the old man told her it was a magic stone",
  "the girl kept the stone in a small box",
  "one day the king rode to the forest",
  "he found a white horse by the lake",
  "the horse led him to a cave in the hill",
  "in the cave there was a chest of gold",
  "the king brought the gold back to the castle",
  "the queen planted roses in the garden",
  "the boy and the girl played by the stream",
  "the brave fox led them through the dark forest",
  "at last they came to a bridge over the river",
  "they crossed the bridge and found a new land",
];

export const CORPUS_PRESETS = {
  simple:  { label: "Simple (6 sentences)",    sentences: SIMPLE_SENTENCES },
  stories: { label: "Stories (100 sentences)", sentences: EXPANDED_STORIES },
};

export function computeCorpusStats(sentences) {
  const allWords = sentences.flatMap((s) => s.split(/\s+/));
  const uniqueWords = new Set(allWords);
  return {
    sentenceCount: sentences.length,
    totalWords: allWords.length,
    uniqueWords: uniqueWords.size,
    avgLength: +(allWords.length / sentences.length).toFixed(1),
  };
}

// ── Word categories for visualization ───────────────────────────────────────

const KNOWN_CATEGORIES = {
  // Animals
  cat: "animal", dog: "animal", bird: "animal", fish: "animal",
  horse: "animal", mouse: "animal", rabbit: "animal", frog: "animal",
  duck: "animal", fox: "animal", bear: "animal", deer: "animal",
  owl: "animal", wolf: "animal", sheep: "animal", hen: "animal",

  // People
  boy: "person", girl: "person", man: "person", woman: "person",
  king: "person", queen: "person", child: "person", farmer: "person",
  baker: "person", teacher: "person",

  // Objects
  mat: "object", rug: "object", bone: "object", book: "object",
  ball: "object", cup: "object", hat: "object", box: "object",
  boat: "object", cart: "object", door: "object", bed: "object",
  cake: "object", bell: "object", castle: "object", house: "object",
  store: "object", school: "object", barn: "object", bridge: "object",
  nest: "object", bench: "object", shelf: "object", chest: "object",
  throne: "object", cape: "object", key: "object", road: "object",
  wall: "object", yard: "object", cave: "object", bread: "object",
  star: "object", corn: "object", roses: "object",

  // Nature
  sun: "nature", moon: "nature", rain: "nature", wind: "nature",
  tree: "nature", trees: "nature", river: "nature", lake: "nature", hill: "nature",
  field: "nature", sky: "nature", snow: "nature", cloud: "nature",
  leaf: "nature", leaves: "nature", flower: "nature", flowers: "nature",
  stone: "nature", park: "nature", forest: "nature", garden: "nature",
  grass: "nature", pond: "nature", bush: "nature", ground: "nature",
  stream: "nature", frost: "nature", clouds: "nature",

  // Verbs
  sat: "verb", ate: "verb", is: "verb", ran: "verb", went: "verb",
  came: "verb", saw: "verb", had: "verb", was: "verb", got: "verb",
  read: "verb", made: "verb", fell: "verb", put: "verb", let: "verb",
  chased: "verb", walked: "verb", jumped: "verb", swam: "verb",
  flew: "verb", sang: "verb", found: "verb", gave: "verb",
  took: "verb", played: "verb", liked: "verb", wanted: "verb",
  rose: "verb", blew: "verb", shone: "verb", hid: "verb",
  stood: "verb", fed: "verb", sold: "verb", set: "verb",
  grew: "verb", turned: "verb", covered: "verb", drifted: "verb",
  led: "verb", crossed: "verb", brought: "verb", planted: "verb",
  kept: "verb", told: "verb", rode: "verb",

  // Adjectives
  big: "adjective", small: "adjective", tall: "adjective", old: "adjective",
  new: "adjective", good: "adjective", cold: "adjective", hot: "adjective",
  red: "adjective", blue: "adjective", green: "adjective", brown: "adjective",
  white: "adjective", black: "adjective", long: "adjective", dark: "adjective",
  bright: "adjective", deep: "adjective", little: "adjective", brave: "adjective",
  warm: "adjective", fresh: "adjective", wise: "adjective", magic: "adjective",
  still: "adjective", full: "adjective", gold: "adjective",

  // Grammar / function words
  the: "grammar", on: "grammar", a: "grammar", an: "grammar",
  in: "grammar", to: "grammar", for: "grammar", with: "grammar",
  from: "grammar", by: "grammar", of: "grammar", up: "grammar",
  are: "grammar", were: "grammar", and: "grammar", but: "grammar",
  or: "grammar", not: "grammar", at: "grammar", it: "grammar",
  he: "grammar", she: "grammar", they: "grammar", him: "grammar", his: "grammar",
  her: "grammar", its: "grammar", there: "grammar", then: "grammar",
  once: "grammar", upon: "grammar", time: "grammar", across: "grammar",
  into: "grammar", over: "grammar", under: "grammar", near: "grammar",
  very: "grammar", as: "grammar", off: "grammar", down: "grammar",
  through: "grammar", home: "grammar", back: "grammar", day: "grammar",
  night: "grammar", one: "grammar", them: "grammar", last: "grammar",
  end: "grammar", town: "grammar", land: "grammar", room: "grammar",
  north: "grammar", south: "grammar", east: "grammar", west: "grammar",
  class: "grammar", fire: "grammar", quest: "grammar", feast: "grammar",
};

export const CATEGORY_COLORS = {
  animal:    "#4ade80",  // green
  person:    "#f472b6",  // pink
  object:    "#fb923c",  // orange
  nature:    "#34d399",  // emerald
  verb:      "#60a5fa",  // blue
  adjective: "#fbbf24",  // amber
  grammar:   "#94a3b8",  // slate
  other:     "#e879f9",  // fuchsia
  subword:   "#c084fc",  // purple
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

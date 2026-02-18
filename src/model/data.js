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
};

export function buildVocabulary(sentences) {
  const allWords = [...new Set(sentences.flatMap((s) => s.split(" ")))];
  const word2id = {};
  allWords.forEach((w, i) => (word2id[w] = i));

  const wordCategories = {};
  allWords.forEach((w) => {
    wordCategories[w] = KNOWN_CATEGORIES[w] || "other";
  });

  return {
    allWords,
    word2id,
    id2word: allWords,
    vocabSize: allWords.length,
    wordCategories,
  };
}

export function makeTrainingData(sentences, word2id) {
  const data = [];
  for (const sent of sentences) {
    const tokens = sent.split(" ").map((w) => word2id[w]);
    for (let i = 1; i < tokens.length; i++) {
      const inputSeq = tokens.slice(0, i);
      data.push({ input: inputSeq, target: tokens[i], sentence: sent });
    }
  }
  return data;
}

export function generatePrompts(sentences, word2id) {
  const prompts = [];
  const seen = new Set();
  for (const sent of sentences) {
    const words = sent.split(" ");
    if (words.length < 2) continue;
    const inputWords = words.slice(0, -1);
    const target = words[words.length - 1];
    const key = inputWords.join(" ");
    if (seen.has(key)) continue;
    seen.add(key);
    const tokens = inputWords.map((w) => word2id[w]);
    prompts.push({ tokens, label: `${inputWords.join(" ")} ___`, target });
  }
  return prompts;
}

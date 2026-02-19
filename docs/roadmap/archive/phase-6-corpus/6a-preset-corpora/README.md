# 5A: Preset Corpora

## File
`src/model/data.js`

## Corpora

### `SIMPLE_SENTENCES` (current, backward compatible)
The existing 6 sentences:
```
the cat sat on the mat
the dog sat on the rug
the cat ate the fish
the dog ate the bone
the cat is on the mat
the dog is on the rug
```

### `EXPANDED_STORIES` (~100-150 sentences)
Organized into thematic groups:

**Animals & Actions (~25 sentences)**
- "the big cat chased the small mouse"
- "the brown dog ran to the park"
- "a bird sat on the tall tree"
- etc.

**People & Daily Life (~25 sentences)**
- "the boy went to the school"
- "the girl read a good book"
- "the man walked to the store"
- etc.

**Nature & Weather (~25 sentences)**
- "the sun rose in the east"
- "the rain fell on the trees"
- "a cold wind blew from the north"
- etc.

**Simple Narratives (~25 sentences)**
- "once upon a time there was a king"
- "the king had a big castle"
- "the queen sat in the garden"
- etc.

### Design Principles
- Simple, declarative sentences (subject-verb-object pattern)
- Controlled vocabulary: ~150-250 unique words
- Overlapping words across themes (the, a, on, in, to, was, is)
- Enough variety for BPE to produce meaningful subword merges
- Short enough to fit in seqLen=32 after tokenization

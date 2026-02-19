# 8B: Embedding Panel — Hover Labels + Zoom

## File
`src/components/EmbeddingPanel.jsx`

## Problem
Currently renders every word as a labeled dot. With 200 tokens, labels overlap and the plot becomes unreadable.

## Solutions

### Conditional Labels
- **vocabSize <= 30**: Show labels always (current behavior)
- **vocabSize > 30**: Hide labels by default, show on hover only

### Hover Interaction
- On mouse over a dot: show tooltip with token name, category, and embedding norm
- Highlight related tokens (same category) on hover

### Visual Differentiation
- Full words: circles
- BPE subword tokens: diamonds or smaller dots
- Opacity by token frequency in corpus (more frequent = more opaque)

### Category Filter
- Small dropdown or toggle buttons for categories
- Click a category to show only that category's dots (others fade to 10% opacity)
- "All" button to reset

### Performance
- SVG with 200 circles is fine (no need for canvas at this scale)
- PCA computation already memoized with `useMemo`

# 8A: Prediction Panel — Top-K View

## File
`src/components/PredictionPanel.jsx`

## Problem
Currently shows ALL vocabulary words as bars. With 200 BPE tokens, the panel becomes an unusable wall of tiny bars.

## Solution: Top-K Display

### Default View
- Show **top 10** predictions as bars (sorted by probability)
- If target word is not in top 10, show it separately below with a dashed border
- "Show all" toggle expands to full list (scrollable)

### Visual Tweaks
- BPE subword tokens: show with a subtle marker (e.g., leading dot) to distinguish from full words
- Color coding: use `CATEGORY_COLORS.subword` (#c084fc purple) for subword tokens
- Probability threshold: tokens below 1% shown at minimum bar width with faded text

### Performance
- Only render top-K bars by default (10 DOM elements instead of 200)
- Full list uses virtual scrolling or simple overflow-y: auto

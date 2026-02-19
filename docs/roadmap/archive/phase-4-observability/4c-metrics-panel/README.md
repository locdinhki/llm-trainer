# 4C: Metrics Panel

## New File: `src/components/MetricsPanel.jsx`

## UI Layout
```
+------------------+------------------+------------------+
| PERPLEXITY       | GRADIENT NORM    | LEARNING RATE    |
| ~~~~~~~~\        |    /\   /\       | /~~~\            |
|          \____   |   /  \_/  \__    |      \________   |
| --- vocabSize    |   --- clip=5.0   | --- theoretical  |
| Last: 2.31       |   Last: 0.42     | Current: 0.0008  |
+------------------+------------------+------------------+
```

## Structure
Full-width panel (gridColumn: "1 / -1") with 3 mini SVG charts in a 1fr 1fr 1fr CSS grid.

Each mini-chart follows the LossPanel SVG pattern:
- Dark card background (`rgba(255,255,255,0.03)`)
- JetBrains Mono uppercase header with letter-spacing
- Manual Y-axis scaling (minVal, maxVal, range)
- SVG path for data line + dashed reference line

## Chart Specifications

### Perplexity (left)
- Color: `#a78bfa` (purple)
- Reference line: `vocabSize` (random guessing baseline), dashed orange
- Y-axis: log scale recommended for large ranges
- Label: last value

### Gradient Norm (center)
- Color: `#22d3ee` (cyan)
- Reference line: gradient clip threshold (5.0), dashed red
- Spikes indicate training instability
- Label: last value

### Learning Rate (right)
- Color: `#60a5fa` (blue)
- Theoretical curve: precomputed from schedule params, dashed gray overlay
- Dot marker at current step position
- Label: current LR value

## Props
```jsx
MetricsPanel({
  perplexityHistory,  // derived from lossHistory
  gradNormHistory,    // from training loop
  lrHistory,          // from training loop
  vocabSize,          // for perplexity reference
  currentStep,
  warmupSteps,
  totalSteps,
  baseLR,
  useLRSchedule,
})
```

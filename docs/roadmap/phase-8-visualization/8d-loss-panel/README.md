# 8D: Loss Panel — Perplexity + Smoothing

## File
`src/components/LossPanel.jsx`

## Enhancements

### Perplexity Display
- Show perplexity alongside loss in the panel header
- `perplexity = exp(loss)`
- More intuitive metric: "the model is choosing between N equally likely tokens"
- Example: loss=2.3 -> perplexity=10 -> model is as uncertain as random guessing among 10 tokens

### Smoothed Loss Line
- Add exponential moving average overlay on the loss chart
- `smoothed = alpha * loss + (1 - alpha) * prev_smoothed` (alpha = 0.1)
- Solid line for smoothed, faded dots/thin line for raw loss
- Helps see the trend through mini-batch noise

### Increased History Buffer
- Current: 500 data points
- New: 2,000 data points
- Training may run for 5,000+ steps with larger models
- Downsample for display if needed (every Nth point for the chart)

### Visual Polish
- Show epoch markers on x-axis (vertical dashed lines)
- Current learning rate displayed in header (if using LR schedule)
- Random baseline line already exists — update for new vocab sizes

# 8C: Attention Panel — Canvas Renderer

## File
`src/components/AttentionPanel.jsx`

## Problem
Currently renders heatmap as DOM divs. At 32 tokens: 32x32 = 1,024 cells per heatmap. With 12 heatmaps visible (3 blocks x 4 heads), that's 12,288 DOM elements — too many.

## Solutions

### Canvas Renderer (seqLen > 16)
- Switch from DOM divs to `<canvas>` 2D rendering
- Draw colored rectangles for each cell
- Much faster: single DOM element regardless of grid size
- Token labels drawn as canvas text along edges

### Overview Mode
- Small grid showing ALL block x head heatmaps as thumbnails
- Layout: 3 rows (blocks) x 4 cols (heads)
- Each thumbnail is ~60x60 pixels
- Click a thumbnail to expand to full detail view

### Detail View
- Full-size heatmap for selected block + head (current behavior, but on canvas)
- Click a query row to see its attention distribution as a bar chart
- Hover to see exact attention weight values

### Token Labels for BPE
- BPE tokens may be subword pieces like "th", "at", "##ing"
- Show decoded token strings as labels
- Truncate long tokens with ellipsis
- Rotate labels if needed for long sequences

### Performance Target
- Canvas draw: < 5ms for 32x32 heatmap
- Overview grid: < 10ms total for 12 thumbnails
- Smooth 60fps during training updates

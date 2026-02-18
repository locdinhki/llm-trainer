# 6A: Expanded Config Options

## File
`src/components/SettingsPanel.jsx`

## New/Expanded Controls

### Model Architecture
| Setting | Current Options | New Options |
|---------|----------------|-------------|
| embedDim | [8, 16, 32] | [16, 32, 48, 64] |
| numBlocks | [1, 2, 3, 4] | [1, 2, 3, 4, 6] |
| numHeads | [1, 2, 4] | Dynamic: valid divisors of embedDim |
| ffnDim | multiplier 1-4x | [2x, 3x, 4x] of embedDim |
| seqLen | slider 4-20 | [10, 16, 24, 32, 48] |

### Training Options
| Setting | Description |
|---------|-------------|
| Activation | Segmented: ReLU / GELU / SiLU |
| Dropout | Slider: 0.0 to 0.3 (step 0.05) |
| Optimizer | Segmented: SGD / Adam |
| Batch size | Segmented: Full / 16 / 32 / 64 |

### Preset Configs
Quick-select buttons that set multiple values at once:

| Preset | embedDim | blocks | heads | ffnDim | ~Params |
|--------|----------|--------|-------|--------|---------|
| Tiny | 16 | 1 | 2 | 32 | ~3K |
| Small | 32 | 2 | 4 | 96 | ~30K |
| Medium | 48 | 3 | 4 | 128 | ~90K |
| Large | 64 | 4 | 8 | 256 | ~300K |

### Info Display
- Estimated parameter count (computed live from config)
- Estimated ms/step (rough calculation based on config)
- Warning for "Large" preset: "Training may be slow"

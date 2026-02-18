# Tiny Transformer — Live Training Dashboard

An interactive React dashboard that implements a complete transformer language model in pure JavaScript and trains it live in the browser. Built as an educational tool to visually demonstrate how LLMs learn, step by step.

![Dark themed dashboard with 4 visualization panels]

---

## What This Is

This is a **from-scratch transformer** — the same architecture behind GPT, Llama, and Claude — shrunk down to ~13,000 parameters so you can watch every piece of the learning process in real time. No pre-trained weights, no external ML libraries. Just math running in your browser.

The model learns to complete simple sentences:

| Input | Expected Prediction |
|---|---|
| `the cat sat on the ___` | mat |
| `the dog sat on the ___` | rug |
| `the cat ate the ___` | fish |
| `the dog ate the ___` | bone |

Before training, it guesses randomly (~10% per word). After training, it should confidently predict the correct word at 80%+ accuracy.

---

## Architecture

```
Input Tokens
     ↓
Token Embedding (10 × 16)     ← learnable lookup table
     +
Positional Embedding (10 × 16) ← encodes word position
     ↓
┌─────────────────────────────┐
│     Transformer Block        │
│  ┌───────────────────────┐  │
│  │   Multi-Head Attention │  │  2 heads, 8 dims each
│  │   (Causal Masked)      │  │  Query, Key, Value projections
│  └───────────────────────┘  │
│          + Residual          │
│  ┌───────────────────────┐  │
│  │   Feed-Forward Network │  │  16 → 32 → 16 (ReLU)
│  └───────────────────────┘  │
│          + Residual          │
│       Layer Norm (×2)        │
└─────────────────────────────┘
     ↓
Output Projection (16 → 10)
     ↓
Softmax → Probabilities
```

### Configuration

| Parameter | Value | Notes |
|---|---|---|
| Vocabulary size | 10 | Unique words in training data |
| Embedding dimension | 16 | Vector size per token |
| Attention heads | 2 | Each head has 8 dimensions |
| FFN hidden dimension | 32 | 2× embedding dim |
| Max sequence length | 10 | Positional embedding slots |
| Transformer blocks | 1 | Single layer (expandable) |

### Parameter Count Breakdown

| Component | Parameters |
|---|---|
| Token embeddings | 10 × 16 = 160 |
| Positional embeddings | 10 × 16 = 160 |
| Attention (Wq, Wk, Wv, Wo + biases) | 4 × (16×16 + 16) = 1,088 |
| Layer Norm 1 (γ, β) | 2 × 16 = 32 |
| FFN (W1, b1, W2, b2) | 16×32 + 32 + 32×16 + 16 = 1,072 |
| Layer Norm 2 (γ, β) | 2 × 16 = 32 |
| Output projection + bias | 16×10 + 10 = 170 |
| **Total** | **~2,714** |

---

## Dashboard Panels

### 1. Prediction Probabilities (Top-Left)

The most important panel. Shows the model's probability distribution over all vocabulary words for the currently selected prompt.

**What to watch for:**
- **Step 0:** All words at roughly 10% (uniform random — the model knows nothing)
- **Early training:** Probabilities shift chaotically as gradients push weights around
- **Mid training:** The correct answer starts pulling ahead of competitors
- **Converged:** Correct word at 80%+ with a dominant green bar

Words are sorted by probability, color-coded by category (animals in green, objects in orange, verbs in blue, grammar words in gray). The correct target word is highlighted.

### 2. Embedding Space (Top-Right)

A 2D PCA projection of all word embeddings. Each dot is a word, positioned by how the model currently represents it internally.

**What to watch for:**
- **Step 0:** Random scatter — "cat" might be next to "mat" by accident
- **During training:** Similar words drift toward each other
- **Converged:** Clear clusters form — animals together, objects together, verbs together

This makes the abstract concept of "word vectors" tangible. The model literally learns that "cat" and "dog" behave similarly by placing them near each other in this space.

### 3. Attention Heatmap (Bottom-Left)

Shows which input words the model "looks at" when making its prediction. Brighter cells = higher attention weight. Toggle between Head 1 and Head 2 to see different attention patterns.

**What to watch for:**
- **Step 0:** Uniform attention (every word attends equally to every previous word)
- **During training:** Patterns emerge — some heads learn to attend to the subject ("cat" vs "dog"), others attend to the verb or preposition
- **Converged:** Sharp, meaningful patterns — e.g., the last position strongly attends to the subject word

The causal mask is visible (dark triangle in upper-right) — each word can only attend to words before it, not future words. This is how GPT-style models work.

### 4. Loss Curve (Bottom-Right)

Cross-entropy loss over training steps. The dashed orange line marks "random guess" baseline (−ln(1/10) ≈ 2.3).

**What to watch for:**
- **Start:** Loss near 2.3 (random guessing)
- **Learning:** Loss drops, sometimes with bumps (the model briefly gets confused as it reorganizes weights)
- **Converged:** Loss approaches 0 (the model is confident and correct)

---

## Controls

| Control | Function |
|---|---|
| **▶ Step** | Run one training step manually. Best for narration — click and explain what changed. |
| **⏵ Play / ⏸ Pause** | Auto-advance through training steps continuously. |
| **Speed (1–10×)** | Controls delay between auto-play steps. |
| **↺ Reset** | Reinitialize all weights to random. Start fresh. |
| **Prompt buttons** | Switch which sentence the model is being tested on. All 4 prompts are always trained simultaneously — this just changes which one is visualized. |

---

## How Training Works (Under the Hood)

### Forward Pass

1. Convert words to token IDs via vocabulary lookup
2. Look up token embeddings + add positional embeddings
3. Pass through self-attention (multi-head, causally masked)
4. Pass through feed-forward network with ReLU activation
5. Project final hidden state to vocabulary-sized logits
6. Softmax to get probabilities

### Loss Computation

Cross-entropy loss: `−log(P(correct_word))`

If the model assigns 10% probability to the correct word, loss = −log(0.1) = 2.3. If it assigns 90%, loss = −log(0.9) = 0.105. The model is incentivized to put maximum probability on the correct answer.

### Gradient Computation

This implementation uses **numerical (finite-difference) gradients** instead of backpropagation:

```
gradient ≈ (Loss(weight + ε) − Loss(weight)) / ε
```

For each parameter, we nudge it slightly, measure how loss changes, and use that to determine which direction to update. This is conceptually clear (you can see exactly what's happening) but computationally expensive — real frameworks use automatic differentiation (backprop) which is mathematically equivalent but orders of magnitude faster.

### Weight Update

Simple gradient descent: `weight = weight − learning_rate × gradient`

Learning rate is set to 0.01. Too high and the model oscillates. Too low and it learns too slowly.

---

## Training Data

Six sentences generate 25 training examples (each subsequence predicts the next word):

```
"the cat sat on the mat"     →  5 training pairs
"the dog sat on the rug"     →  5 training pairs
"the cat ate the fish"       →  4 training pairs
"the dog ate the bone"       →  4 training pairs
"the cat is on the mat"      →  5 training pairs
"the dog is on the rug"      →  5 training pairs
```

Example training pairs from "the cat sat on the mat":

| Input | Target |
|---|---|
| `[the]` | cat |
| `[the, cat]` | sat |
| `[the, cat, sat]` | on |
| `[the, cat, sat, on]` | the |
| `[the, cat, sat, on, the]` | mat |

---

## Vocabulary

| Token ID | Word | Category | Color |
|---|---|---|---|
| 0 | the | grammar | gray |
| 1 | cat | animal | green |
| 2 | sat | verb | blue |
| 3 | on | grammar | gray |
| 4 | mat | object | orange |
| 5 | dog | animal | green |
| 6 | rug | object | orange |
| 7 | ate | verb | blue |
| 8 | fish | object | orange |
| 9 | bone | object | orange |
| 10 | is | verb | blue |

---

## Scaling Context

This toy model and production LLMs use the **exact same architecture**. The only difference is scale:

| | This Model | GPT-3 | GPT-4 (est.) |
|---|---|---|---|
| Parameters | ~2,700 | 175 billion | ~1.8 trillion |
| Vocabulary | 10 words | 50,257 tokens | 100,000+ tokens |
| Embedding dim | 16 | 12,288 | ~16,000+ |
| Attention heads | 2 | 96 | 120+ |
| Transformer blocks | 1 | 96 | 120+ |
| Training data | 6 sentences | 300B tokens | 13T+ tokens |
| Training time | Seconds | Weeks on 1000s of GPUs | Months |
| Training cost | $0 | ~$4.6 million | ~$100+ million |

---

## File Structure

```
llm-training-dashboard.jsx    ← The complete application (single file)
README.md                     ← This file
```

The entire implementation — transformer, training loop, PCA visualization, and dashboard UI — lives in one self-contained React JSX file with no external ML dependencies.

---

## Technical Notes

- **Pure JavaScript ML:** No TensorFlow, PyTorch, or ONNX. Every matrix multiply, softmax, and layer norm is hand-written for transparency.
- **Numerical gradients:** Intentionally chosen over backprop for clarity. Each gradient computation requires `2 × num_parameters` forward passes per step, making it slow but inspectable.
- **PCA implementation:** Power iteration method for computing top-2 principal components, with Gram-Schmidt orthogonalization. No external linear algebra library.
- **Causal masking:** Future tokens are masked with −∞ before softmax, preventing the model from "cheating" by looking ahead.
- **Seeded randomness:** Uses Mulberry32 PRNG with seed 42 for reproducible initialization. Reset always returns to the same starting state.

---

## Potential Enhancements

- **Multiple transformer blocks:** Add 2–4 blocks to show how deeper models refine representations layer by layer
- **Training speed:** Implement backpropagation for faster training (100–1000× speedup)
- **Temperature slider:** Control sampling randomness in predictions
- **Weight matrix visualizations:** Heatmaps showing raw parameter values and how they change
- **Custom sentences:** Let users add their own training data
- **Export snapshots:** Save embedding positions and attention maps as images for video editing
- **Comparison mode:** Side-by-side before/after for any training milestone

---

## License

Educational use. Built for learning and demonstration purposes.
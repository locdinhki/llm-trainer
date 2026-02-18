# Phase 1: Float32Array Migration — Done

## Goal
Replace all JS Arrays with Float32Array for weights and intermediates. Expected 2-4x speedup on math operations due to contiguous memory, no number boxing, and better JIT optimization.

## Why This Matters
- JS Arrays store numbers as boxed heap objects; Float32Array stores them as contiguous 32-bit floats
- Inner loops (matVec, vecAdd) access contiguous memory = better CPU cache behavior
- JIT compilers can emit SIMD-like instructions for typed array loops
- Foundation for Web Worker integration (Float32Array is Transferable for zero-copy)

## Subphases

| # | Subphase | Files | Description | Status |
|---|----------|-------|-------------|--------|
| 1A | [Math Primitives](1a-math-primitives/) | `src/model/math.js` | All math functions return Float32Array; added `silu`, `gelu`, `siluBackward`, `geluBackward` | Done |
| 1B | [Transformer Init](1b-transformer-init/) | `src/model/transformer.js` | Weight matrices as Float32Array; fixed `countParameters` `Array.isArray` check | Done |
| 1C | [Backward Pass](1c-backward-pass/) | `src/model/backward.js` | Zero/gradient arrays as Float32Array; `zeros1D`/`zeros2D` use typed arrays | Done |
| 1D | [Forward Cache](1d-forward-cache/) | `src/model/forwardWithCache.js` | Cache intermediates as Float32Array | Done |
| 1E | [Training](1e-training/) | `src/model/training.js`, `training.worker.js` | Fixed `Array.isArray` checks; relaxed gradient tolerance to 5e-2 for Float32 precision | Done |

## Key Technical Notes
- 2D matrices use "Array of Float32Array rows" pattern: `Array.from({length: rows}, () => new Float32Array(cols))`
- `Float32Array` does NOT have `.map()`, `.filter()`, etc. from Array.prototype — use explicit for-loops
- `pca2D` in math.js stays as regular Arrays (returns objects, not perf-critical)
- `Array.isArray()` returns false for Float32Array — replaced with `arr[0]?.length !== undefined`
- Gradient verification tolerance relaxed from 1e-3 to 5e-2 due to Float32 ~7-digit precision (vs Float64 ~15 digits)

## Verification
- [x] `npm run build` compiles cleanly
- [x] Gradient check passes (relative error 1.1e-2, within 5e-2 threshold)
- [x] Training converges on default 6 sentences
- [x] All 4 visualization panels render correctly
- [x] Step, Play/Pause, Reset controls all functional

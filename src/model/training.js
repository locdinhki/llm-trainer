import { forward } from "./transformer.js";
import { softmax } from "./math.js";
import { forwardWithCache } from "./forwardWithCache.js";
import { backward, createZeroGrads } from "./backward.js";

// --- Gradient clipping ---

function clipGrad(grad, maxNorm = 5.0) {
  return Math.max(-maxNorm, Math.min(maxNorm, grad));
}

// --- Loss computation (for evaluation, not training) ---

export function computeLoss(params, trainingData, config) {
  let loss = 0;
  for (const { input, target } of trainingData) {
    const { logits } = forward(params, input, config);
    const probs = softmax(logits);
    loss -= Math.log(probs[target] + 1e-10);
  }
  return loss / trainingData.length;
}

// --- Backpropagation-based training step ---

function scaleGrads(grads, scale) {
  // Scale 2D arrays
  for (const key of ["embedding", "posEmbedding", "Wout"]) {
    const arr = grads[key];
    for (let i = 0; i < arr.length; i++) {
      for (let j = 0; j < arr[i].length; j++) {
        arr[i][j] *= scale;
      }
    }
  }
  // Scale 1D arrays
  for (let i = 0; i < grads.bout.length; i++) {
    grads.bout[i] *= scale;
  }
  // Scale per-block
  for (const gBlock of grads.blocks) {
    for (const key of Object.keys(gBlock)) {
      const arr = gBlock[key];
      if (arr[0]?.length !== undefined) {
        for (let i = 0; i < arr.length; i++) {
          for (let j = 0; j < arr[i].length; j++) {
            arr[i][j] *= scale;
          }
        }
      } else {
        for (let i = 0; i < arr.length; i++) {
          arr[i] *= scale;
        }
      }
    }
  }
}

function applyGradients(params, grads, lr, maxNorm) {
  // Apply to 2D params
  for (const key of ["embedding", "posEmbedding", "Wout"]) {
    const p = params[key];
    const g = grads[key];
    for (let i = 0; i < p.length; i++) {
      for (let j = 0; j < p[i].length; j++) {
        p[i][j] -= lr * clipGrad(g[i][j], maxNorm);
      }
    }
  }
  // Apply to 1D params
  for (let i = 0; i < params.bout.length; i++) {
    params.bout[i] -= lr * clipGrad(grads.bout[i], maxNorm);
  }
  // Apply to per-block params
  for (let b = 0; b < params.blocks.length; b++) {
    const pBlock = params.blocks[b];
    const gBlock = grads.blocks[b];
    for (const key of Object.keys(pBlock)) {
      const p = pBlock[key];
      const g = gBlock[key];
      if (p[0]?.length !== undefined) {
        for (let i = 0; i < p.length; i++) {
          for (let j = 0; j < p[i].length; j++) {
            p[i][j] -= lr * clipGrad(g[i][j], maxNorm);
          }
        }
      } else {
        for (let i = 0; i < p.length; i++) {
          p[i] -= lr * clipGrad(g[i], maxNorm);
        }
      }
    }
  }
}

export function trainStepBackprop(params, trainingData, config, lr) {
  const grads = createZeroGrads(params, config);
  let totalLoss = 0;

  for (const { input, target } of trainingData) {
    const { logits, cache } = forwardWithCache(params, input, config);
    const probs = softmax(logits);
    totalLoss -= Math.log(probs[target] + 1e-10);

    backward(params, cache, target, config, grads);
  }

  totalLoss /= trainingData.length;
  scaleGrads(grads, 1 / trainingData.length);
  applyGradients(params, grads, lr, 5.0);

  return totalLoss;
}

// --- Numerical gradient verification (for debugging) ---

export function verifyGradients(params, trainingData, config) {
  const eps = 1e-4;
  const grads = createZeroGrads(params, config);

  for (const { input, target } of trainingData) {
    const { cache } = forwardWithCache(params, input, config);
    backward(params, cache, target, config, grads);
  }
  scaleGrads(grads, 1 / trainingData.length);

  // Check a few parameters from embedding
  const errors = [];
  const checkParams = [
    { name: "embedding[0][0]", get: () => params.embedding[0][0], set: (v) => { params.embedding[0][0] = v; }, grad: grads.embedding[0][0] },
    { name: "embedding[1][1]", get: () => params.embedding[1][1], set: (v) => { params.embedding[1][1] = v; }, grad: grads.embedding[1][1] },
    { name: "Wout[0][0]", get: () => params.Wout[0][0], set: (v) => { params.Wout[0][0] = v; }, grad: grads.Wout[0][0] },
    { name: "bout[0]", get: () => params.bout[0], set: (v) => { params.bout[0] = v; }, grad: grads.bout[0] },
    { name: "blocks[0].Wq[0][0]", get: () => params.blocks[0].Wq[0][0], set: (v) => { params.blocks[0].Wq[0][0] = v; }, grad: grads.blocks[0].Wq[0][0] },
    { name: "blocks[0].W1[0][0]", get: () => params.blocks[0].W1[0][0], set: (v) => { params.blocks[0].W1[0][0] = v; }, grad: grads.blocks[0].W1[0][0] },
    { name: "blocks[0].ln1_g[0]", get: () => params.blocks[0].ln1_g[0], set: (v) => { params.blocks[0].ln1_g[0] = v; }, grad: grads.blocks[0].ln1_g[0] },
  ];

  for (const { name, get, set, grad: analyticalGrad } of checkParams) {
    const orig = get();
    set(orig + eps);
    const lossPlus = computeLoss(params, trainingData, config);
    set(orig - eps);
    const lossMinus = computeLoss(params, trainingData, config);
    set(orig);
    const numericalGrad = (lossPlus - lossMinus) / (2 * eps);

    const absDiff = Math.abs(analyticalGrad - numericalGrad);
    const relError = absDiff / (Math.abs(analyticalGrad) + Math.abs(numericalGrad) + 1e-10);
    errors.push({ name, analytical: analyticalGrad, numerical: numericalGrad, relError });
  }

  const maxRelError = Math.max(...errors.map((e) => e.relError));
  const passed = maxRelError < 5e-2; // relaxed for Float32Array precision (~7 digits vs ~15 for Float64)

  console.log(`Gradient verification: ${passed ? "PASSED" : "FAILED"} (max relative error: ${maxRelError.toExponential(3)})`);
  for (const e of errors) {
    console.log(`  ${e.name}: analytical=${e.analytical.toExponential(4)}, numerical=${e.numerical.toExponential(4)}, relErr=${e.relError.toExponential(3)}${e.relError > 5e-2 ? " *** MISMATCH ***" : ""}`);
  }

  return { passed, maxRelError, errors };
}

// --- Legacy numerical gradient functions (kept for reference) ---

function updateParam(param, params, trainingData, config, lr, eps) {
  if (param[0]?.length !== undefined) {
    for (let i = 0; i < param.length; i++) {
      for (let j = 0; j < param[i].length; j++) {
        const orig = param[i][j];
        param[i][j] = orig + eps;
        const lossPlus = computeLoss(params, trainingData, config);
        param[i][j] = orig - eps;
        const lossMinus = computeLoss(params, trainingData, config);
        param[i][j] = orig;
        const grad = clipGrad((lossPlus - lossMinus) / (2 * eps));
        param[i][j] = orig - lr * grad;
      }
    }
  } else {
    for (let i = 0; i < param.length; i++) {
      const orig = param[i];
      param[i] = orig + eps;
      const lossPlus = computeLoss(params, trainingData, config);
      param[i] = orig - eps;
      const lossMinus = computeLoss(params, trainingData, config);
      param[i] = orig;
      const grad = clipGrad((lossPlus - lossMinus) / (2 * eps));
      param[i] = orig - lr * grad;
    }
  }
}

export function trainStep(params, trainingData, config, lr) {
  const eps = 1e-4;
  const totalLoss = computeLoss(params, trainingData, config);
  const sharedKeys = ["embedding", "posEmbedding", "Wout", "bout"];
  for (const key of sharedKeys) {
    updateParam(params[key], params, trainingData, config, lr, eps);
  }
  const blockKeys = [
    "Wq", "Wk", "Wv", "Wo", "bq", "bk", "bv", "bo",
    "W1", "W2", "b1", "b2", "ln1_g", "ln1_b", "ln2_g", "ln2_b",
  ];
  for (const block of params.blocks) {
    for (const key of blockKeys) {
      updateParam(block[key], params, trainingData, config, lr, eps);
    }
  }
  return totalLoss;
}

export function trainStepFast(params, trainingData, config, lr) {
  const eps = 1e-4;
  const totalLoss = computeLoss(params, trainingData, config);
  for (const key of ["embedding", "Wout", "bout"]) {
    updateParam(params[key], params, trainingData, config, lr, eps);
  }
  const blockKeys = ["Wq", "Wk", "Wv", "Wo", "W1", "W2", "b1", "b2"];
  for (const block of params.blocks) {
    for (const key of blockKeys) {
      updateParam(block[key], params, trainingData, config, lr, eps);
    }
  }
  return totalLoss;
}

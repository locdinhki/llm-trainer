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
  for (const key of ["embedding", "Wout"]) {
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
  for (const key of ["embedding", "Wout"]) {
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

// --- Gradient norm computation ---

function computeGradNorm(grads) {
  let sumSq = 0;
  for (const key of ["embedding", "Wout"]) {
    const arr = grads[key];
    for (let i = 0; i < arr.length; i++) {
      for (let j = 0; j < arr[i].length; j++) {
        sumSq += arr[i][j] * arr[i][j];
      }
    }
  }
  for (let i = 0; i < grads.bout.length; i++) {
    sumSq += grads.bout[i] * grads.bout[i];
  }
  for (const gBlock of grads.blocks) {
    for (const key of Object.keys(gBlock)) {
      const arr = gBlock[key];
      if (arr[0]?.length !== undefined) {
        for (let i = 0; i < arr.length; i++) {
          for (let j = 0; j < arr[i].length; j++) {
            sumSq += arr[i][j] * arr[i][j];
          }
        }
      } else {
        for (let i = 0; i < arr.length; i++) {
          sumSq += arr[i] * arr[i];
        }
      }
    }
  }
  return Math.sqrt(sumSq);
}

// --- 4A: Mini-batch training with AdamW ---

export function trainStepMiniBatch(params, trainingData, config, lr, batchSize, adamState, weightDecay = 0.01) {
  const n = trainingData.length;
  const actualBatch = Math.min(batchSize, n);
  const grads = createZeroGrads(params, config);
  let totalLoss = 0;

  // Random sampling without replacement (Fisher-Yates partial shuffle)
  const indices = new Array(n);
  for (let i = 0; i < n; i++) indices[i] = i;
  for (let i = 0; i < actualBatch; i++) {
    const j = i + Math.floor(Math.random() * (n - i));
    const tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
  }

  for (let i = 0; i < actualBatch; i++) {
    const { input, target } = trainingData[indices[i]];
    const { logits, cache } = forwardWithCache(params, input, config);
    const probs = softmax(logits);
    totalLoss -= Math.log(probs[target] + 1e-10);
    backward(params, cache, target, config, grads);
  }

  totalLoss /= actualBatch;
  scaleGrads(grads, 1 / actualBatch);

  const gradNorm = computeGradNorm(grads);

  if (adamState) {
    adamUpdate(params, grads, adamState, lr, 0.9, 0.999, 1e-8, weightDecay);
  } else {
    applyGradients(params, grads, lr, 5.0);
  }

  return { loss: totalLoss, gradNorm };
}

// --- 4B: AdamW optimizer (decoupled weight decay) ---

export function createAdamState(params, config) {
  return {
    m: createZeroGrads(params, config),
    v: createZeroGrads(params, config),
    t: 0,
  };
}

function adamUpdateArray1D(p, g, m, v, lr, beta1, beta2, eps, mHatScale, vHatScale, maxNorm, weightDecay) {
  for (let i = 0; i < p.length; i++) {
    const gc = clipGrad(g[i], maxNorm);
    m[i] = beta1 * m[i] + (1 - beta1) * gc;
    v[i] = beta2 * v[i] + (1 - beta2) * gc * gc;
    const mHat = m[i] * mHatScale;
    const vHat = v[i] * vHatScale;
    p[i] -= lr * (mHat / (Math.sqrt(vHat) + eps) + weightDecay * p[i]);
  }
}

function adamUpdateArray2D(p, g, m, v, lr, beta1, beta2, eps, mHatScale, vHatScale, maxNorm, weightDecay) {
  for (let i = 0; i < p.length; i++) {
    adamUpdateArray1D(p[i], g[i], m[i], v[i], lr, beta1, beta2, eps, mHatScale, vHatScale, maxNorm, weightDecay);
  }
}

export function adamUpdate(params, grads, state, lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 0.01) {
  state.t += 1;
  const maxNorm = 5.0;
  const mHatScale = 1 / (1 - Math.pow(beta1, state.t));
  const vHatScale = 1 / (1 - Math.pow(beta2, state.t));

  // 2D params
  for (const key of ["embedding", "Wout"]) {
    adamUpdateArray2D(params[key], grads[key], state.m[key], state.v[key], lr, beta1, beta2, eps, mHatScale, vHatScale, maxNorm, weightDecay);
  }
  // 1D params
  adamUpdateArray1D(params.bout, grads.bout, state.m.bout, state.v.bout, lr, beta1, beta2, eps, mHatScale, vHatScale, maxNorm, weightDecay);
  // Per-block params
  for (let b = 0; b < params.blocks.length; b++) {
    const pBlock = params.blocks[b];
    const gBlock = grads.blocks[b];
    const mBlock = state.m.blocks[b];
    const vBlock = state.v.blocks[b];
    for (const key of Object.keys(pBlock)) {
      const p = pBlock[key];
      const g = gBlock[key];
      const mB = mBlock[key];
      const vB = vBlock[key];
      if (p[0]?.length !== undefined) {
        adamUpdateArray2D(p, g, mB, vB, lr, beta1, beta2, eps, mHatScale, vHatScale, maxNorm, weightDecay);
      } else {
        adamUpdateArray1D(p, g, mB, vB, lr, beta1, beta2, eps, mHatScale, vHatScale, maxNorm, weightDecay);
      }
    }
  }
}

// --- 3C: Learning rate schedule (warmup + cosine decay) ---

export function getLearningRate(step, warmupSteps, totalSteps, baseLR) {
  if (step < warmupSteps) {
    return baseLR * (step / warmupSteps);
  }
  const progress = (step - warmupSteps) / (totalSteps - warmupSteps);
  return baseLR * 0.5 * (1 + Math.cos(Math.PI * Math.min(progress, 1)));
}

// --- Numerical gradient verification (for debugging) ---

export function verifyGradients(params, trainingData, config) {
  // Disable dropout for gradient verification (stochastic forward breaks numerical diff)
  const verifyConfig = { ...config, dropout: 0 };
  const eps = 1e-3; // larger eps for Float32Array precision (~7 digits vs ~15 for Float64)
  const grads = createZeroGrads(params, verifyConfig);

  for (const { input, target } of trainingData) {
    const { cache } = forwardWithCache(params, input, verifyConfig);
    backward(params, cache, target, verifyConfig, grads);
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
    { name: "blocks[0].W_gate[0][0]", get: () => params.blocks[0].W_gate[0][0], set: (v) => { params.blocks[0].W_gate[0][0] = v; }, grad: grads.blocks[0].W_gate[0][0] },
    { name: "blocks[0].ln1_g[0]", get: () => params.blocks[0].ln1_g[0], set: (v) => { params.blocks[0].ln1_g[0] = v; }, grad: grads.blocks[0].ln1_g[0] },
  ];

  for (const { name, get, set, grad: analyticalGrad } of checkParams) {
    const orig = get();
    set(orig + eps);
    const lossPlus = computeLoss(params, trainingData, verifyConfig);
    set(orig - eps);
    const lossMinus = computeLoss(params, trainingData, verifyConfig);
    set(orig);
    const numericalGrad = (lossPlus - lossMinus) / (2 * eps);

    const absDiff = Math.abs(analyticalGrad - numericalGrad);
    const relError = absDiff / (Math.abs(analyticalGrad) + Math.abs(numericalGrad) + 1e-10);
    errors.push({ name, analytical: analyticalGrad, numerical: numericalGrad, relError });
  }

  const maxRelError = Math.max(...errors.map((e) => e.relError));
  const passed = maxRelError < 1e-1; // relaxed for Float32Array precision (~7 digits vs ~15 for Float64)

  console.log(`Gradient verification: ${passed ? "PASSED" : "FAILED"} (max relative error: ${maxRelError.toExponential(3)})`);
  for (const e of errors) {
    console.log(`  ${e.name}: analytical=${e.analytical.toExponential(4)}, numerical=${e.numerical.toExponential(4)}, relErr=${e.relError.toExponential(3)}${e.relError > 1e-1 ? " *** MISMATCH ***" : ""}`);
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
  const sharedKeys = ["embedding", "Wout", "bout"];
  for (const key of sharedKeys) {
    updateParam(params[key], params, trainingData, config, lr, eps);
  }
  const blockKeys = [
    "Wq", "Wk", "Wv", "Wo", "bq", "bk", "bv", "bo",
    "W_gate", "b_gate", "W1", "W2", "b1", "b2", "ln1_g", "ln2_g",
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
  const blockKeys = ["Wq", "Wk", "Wv", "Wo", "W_gate", "W1", "W2", "b1", "b2"];
  for (const block of params.blocks) {
    for (const key of blockKeys) {
      updateParam(block[key], params, trainingData, config, lr, eps);
    }
  }
  return totalLoss;
}

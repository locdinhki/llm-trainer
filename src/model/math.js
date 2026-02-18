// Simple seeded random for reproducibility
export function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function matVec(mat, vec) {
  const rows = mat.length;
  const cols = vec.length;
  const result = new Float32Array(rows);
  for (let i = 0; i < rows; i++) {
    let sum = 0;
    const row = mat[i];
    for (let j = 0; j < cols; j++) {
      sum += row[j] * vec[j];
    }
    result[i] = sum;
  }
  return result;
}

export function vecAdd(a, b) {
  const n = a.length;
  const result = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    result[i] = a[i] + b[i];
  }
  return result;
}

export function vecMul(a, b) {
  const n = a.length;
  const result = new Float32Array(n);
  for (let i = 0; i < n; i++) result[i] = a[i] * b[i];
  return result;
}

export function vecAddInPlace(target, source) {
  for (let i = 0; i < target.length; i++) {
    target[i] += source[i];
  }
}

export function layerNorm(x, g, b) {
  const n = x.length;
  let sum = 0, sum2 = 0;
  for (let i = 0; i < n; i++) {
    sum += x[i];
    sum2 += x[i] * x[i];
  }
  const mean = sum / n;
  const variance = sum2 / n - mean * mean;
  const invStd = 1 / Math.sqrt(variance + 1e-5);
  const result = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    result[i] = g[i] * ((x[i] - mean) * invStd) + b[i];
  }
  return result;
}

export function rmsNorm(x, gamma) {
  const n = x.length;
  let sumSq = 0;
  for (let i = 0; i < n; i++) sumSq += x[i] * x[i];
  const rms = Math.sqrt(sumSq / n + 1e-6);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = gamma[i] * (x[i] / rms);
  return out;
}

export function softmax(x) {
  const n = x.length;
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    if (x[i] > max) max = x[i];
  }
  const result = new Float32Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    result[i] = Math.exp(x[i] - max);
    sum += result[i];
  }
  for (let i = 0; i < n; i++) {
    result[i] /= sum;
  }
  return result;
}

export function relu(x) {
  const n = x.length;
  const result = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    result[i] = x[i] > 0 ? x[i] : 0;
  }
  return result;
}

export function silu(x) {
  const n = x.length;
  const result = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    result[i] = x[i] / (1 + Math.exp(-x[i]));
  }
  return result;
}

export function gelu(x) {
  const n = x.length;
  const result = new Float32Array(n);
  const c = Math.sqrt(2 / Math.PI);
  for (let i = 0; i < n; i++) {
    const v = x[i];
    result[i] = 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v * v * v)));
  }
  return result;
}

export function siluBackward(x, dOut) {
  const n = x.length;
  const result = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const sig = 1 / (1 + Math.exp(-x[i]));
    result[i] = dOut[i] * (sig + x[i] * sig * (1 - sig));
  }
  return result;
}

export function geluBackward(x, dOut) {
  const n = x.length;
  const result = new Float32Array(n);
  const c = Math.sqrt(2 / Math.PI);
  for (let i = 0; i < n; i++) {
    const v = x[i];
    const inner = c * (v + 0.044715 * v * v * v);
    const tanhVal = Math.tanh(inner);
    const dtanh = 1 - tanhVal * tanhVal;
    const dinner = c * (1 + 3 * 0.044715 * v * v);
    result[i] = dOut[i] * 0.5 * (1 + tanhVal + v * dtanh * dinner);
  }
  return result;
}

// PCA for 2D embedding visualization
export function pca2D(vectors) {
  const n = vectors.length;
  const d = vectors[0].length;

  // Center the data
  const mean = new Array(d).fill(0);
  for (let vi = 0; vi < n; vi++) {
    const v = vectors[vi];
    for (let i = 0; i < d; i++) mean[i] += v[i] / n;
  }
  const centered = vectors.map((v) => v.map((val, i) => val - mean[i]));

  // Power iteration for top 2 eigenvectors of covariance
  const rand = mulberry32(123);
  let pc1 = Array.from({ length: d }, () => rand() - 0.5);
  let pc2 = Array.from({ length: d }, () => rand() - 0.5);

  for (let iter = 0; iter < 20; iter++) {
    const newPc1 = new Array(d).fill(0);
    for (const v of centered) {
      let dot = 0;
      for (let i = 0; i < d; i++) dot += v[i] * pc1[i];
      for (let i = 0; i < d; i++) newPc1[i] += dot * v[i];
    }
    let norm1 = 0;
    for (let i = 0; i < d; i++) norm1 += newPc1[i] * newPc1[i];
    norm1 = Math.sqrt(norm1) + 1e-10;
    for (let i = 0; i < d; i++) pc1[i] = newPc1[i] / norm1;
  }

  // Gram-Schmidt for pc2
  let dot12 = 0;
  for (let i = 0; i < d; i++) dot12 += pc2[i] * pc1[i];
  for (let i = 0; i < d; i++) pc2[i] -= dot12 * pc1[i];

  for (let iter = 0; iter < 20; iter++) {
    const newPc2 = new Array(d).fill(0);
    for (const v of centered) {
      let dot = 0;
      for (let i = 0; i < d; i++) dot += v[i] * pc2[i];
      for (let i = 0; i < d; i++) newPc2[i] += dot * v[i];
    }
    let d12 = 0;
    for (let i = 0; i < d; i++) d12 += newPc2[i] * pc1[i];
    for (let i = 0; i < d; i++) newPc2[i] -= d12 * pc1[i];
    let norm2 = 0;
    for (let i = 0; i < d; i++) norm2 += newPc2[i] * newPc2[i];
    norm2 = Math.sqrt(norm2) + 1e-10;
    for (let i = 0; i < d; i++) pc2[i] = newPc2[i] / norm2;
  }

  return centered.map((v) => {
    let x = 0, y = 0;
    for (let i = 0; i < d; i++) {
      x += v[i] * pc1[i];
      y += v[i] * pc2[i];
    }
    return { x, y };
  });
}

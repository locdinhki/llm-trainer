// Rotary Position Embeddings (RoPE)
// Encodes position by rotating Q/K vectors. Attention scores naturally
// capture relative position (not absolute), which generalizes better.

export function precomputeFreqs(seqLen, headDim, base = 10000) {
  const halfDim = headDim / 2;
  const cos = new Float32Array(seqLen * halfDim);
  const sin = new Float32Array(seqLen * halfDim);
  for (let pos = 0; pos < seqLen; pos++) {
    for (let i = 0; i < halfDim; i++) {
      const theta = pos * Math.pow(base, -2 * i / headDim);
      cos[pos * halfDim + i] = Math.cos(theta);
      sin[pos * halfDim + i] = Math.sin(theta);
    }
  }
  return { cos, sin, halfDim };
}

export function applyRoPE(vec, position, freqs) {
  const half = freqs.halfDim;
  const out = new Float32Array(half * 2);
  const offset = position * half;
  for (let i = 0; i < half; i++) {
    const c = freqs.cos[offset + i];
    const s = freqs.sin[offset + i];
    out[2 * i]     = vec[2 * i] * c - vec[2 * i + 1] * s;
    out[2 * i + 1] = vec[2 * i] * s + vec[2 * i + 1] * c;
  }
  return out;
}

export function applyRoPEBackward(dVec, position, freqs) {
  // Inverse rotation: transpose of rotation matrix (negate sin terms)
  const half = freqs.halfDim;
  const out = new Float32Array(half * 2);
  const offset = position * half;
  for (let i = 0; i < half; i++) {
    const c = freqs.cos[offset + i];
    const s = freqs.sin[offset + i];
    out[2 * i]     =  dVec[2 * i] * c + dVec[2 * i + 1] * s;
    out[2 * i + 1] = -dVec[2 * i] * s + dVec[2 * i + 1] * c;
  }
  return out;
}

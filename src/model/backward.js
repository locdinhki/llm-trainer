import { softmax, vecAddInPlace, vecMul, siluBackward } from "./math.js";
import { precomputeFreqs, applyRoPEBackward } from "./rope.js";

// --- Backward primitives ---

export function softmaxBackward(probs, dProbs) {
  const n = probs.length;
  let dot = 0;
  for (let i = 0; i < n; i++) dot += probs[i] * dProbs[i];
  const dX = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    dX[i] = probs[i] * (dProbs[i] - dot);
  }
  return dX;
}

export function reluBackward(preAct, dOut) {
  const n = preAct.length;
  const dX = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    dX[i] = preAct[i] > 0 ? dOut[i] : 0;
  }
  return dX;
}

export function rmsNormBackward(x, gamma, dOut) {
  const n = x.length;

  // Recompute forward stats
  let sumSq = 0;
  for (let i = 0; i < n; i++) sumSq += x[i] * x[i];
  const rms = Math.sqrt(sumSq / n + 1e-6);
  const invRms = 1 / rms;

  // xNorm[i] = x[i] / rms
  const xNorm = new Float32Array(n);
  for (let i = 0; i < n; i++) xNorm[i] = x[i] * invRms;

  // dGamma[i] = dOut[i] * xNorm[i]
  const dGamma = new Float32Array(n);
  for (let i = 0; i < n; i++) dGamma[i] = dOut[i] * xNorm[i];

  // dX: chain rule through rmsNorm
  // out[i] = gamma[i] * x[i] / rms
  // dL/dx[i] = gamma[i] * dOut[i] / rms - (x[i] / (n * rms^3)) * sum_j(gamma[j] * dOut[j] * x[j])
  let dotGradX = 0;
  for (let i = 0; i < n; i++) dotGradX += gamma[i] * dOut[i] * x[i];

  const dX = new Float32Array(n);
  const invRms3 = invRms / (rms * rms);
  for (let i = 0; i < n; i++) {
    dX[i] = gamma[i] * dOut[i] * invRms - x[i] * dotGradX * invRms3 / n;
  }

  return { dX, dGamma };
}

// matVec backward: out = mat * vec
// dMat[i][j] = dOut[i] * vec[j]
// dVec[j] = sum_i(mat[i][j] * dOut[i])
export function matVecBackward(mat, vec, dOut) {
  const rows = mat.length;
  const cols = vec.length;

  const dMat = new Array(rows);
  for (let i = 0; i < rows; i++) {
    dMat[i] = new Float32Array(cols);
    for (let j = 0; j < cols; j++) {
      dMat[i][j] = dOut[i] * vec[j];
    }
  }

  const dVec = new Float32Array(cols);
  for (let i = 0; i < rows; i++) {
    const row = mat[i];
    const d = dOut[i];
    for (let j = 0; j < cols; j++) {
      dVec[j] += row[j] * d;
    }
  }

  return { dMat, dVec };
}

// --- Gradient helpers ---

function zeros1D(n) {
  return new Float32Array(n);
}

function zeros2D(rows, cols) {
  return Array.from({ length: rows }, () => new Float32Array(cols));
}

function addMat(target, source) {
  for (let i = 0; i < target.length; i++) {
    for (let j = 0; j < target[i].length; j++) {
      target[i][j] += source[i][j];
    }
  }
}

export function createZeroGrads(params, config) {
  const { vocabSize, embedDim, ffnDim, seqLen, numBlocks = 1 } = config;
  const grads = {
    embedding: zeros2D(vocabSize, embedDim),
    Wout: zeros2D(vocabSize, embedDim),
    bout: zeros1D(vocabSize),
    blocks: [],
  };
  for (let b = 0; b < numBlocks; b++) {
    grads.blocks.push({
      Wq: zeros2D(embedDim, embedDim),
      Wk: zeros2D(embedDim, embedDim),
      Wv: zeros2D(embedDim, embedDim),
      Wo: zeros2D(embedDim, embedDim),
      bq: zeros1D(embedDim),
      bk: zeros1D(embedDim),
      bv: zeros1D(embedDim),
      bo: zeros1D(embedDim),
      ln1_g: zeros1D(embedDim),
      W_gate: zeros2D(ffnDim, embedDim),
      b_gate: zeros1D(ffnDim),
      W1: zeros2D(ffnDim, embedDim),
      b1: zeros1D(ffnDim),
      W2: zeros2D(embedDim, ffnDim),
      b2: zeros1D(embedDim),
      ln2_g: zeros1D(embedDim),
    });
  }
  return grads;
}

// --- Main backward function ---

export function backward(params, cache, target, config, grads) {
  const { embedDim, numHeads, numBlocks = 1 } = config;
  const headDim = embedDim / numHeads;
  const len = cache.tokens.length;
  const freqs = precomputeFreqs(len, headDim);

  // 1. Loss gradient: dLogits = softmax(logits) - one_hot(target)
  const probs = softmax(cache.logits);
  const dLogits = new Float32Array(probs.length);
  for (let i = 0; i < probs.length; i++) {
    dLogits[i] = probs[i] - (i === target ? 1 : 0);
  }

  // 2. Output projection backward: logits = matVec(Wout, lastHidden) + bout
  const { dMat: dWout, dVec: dLastHidden } = matVecBackward(
    params.Wout, cache.lastHidden, dLogits
  );
  addMat(grads.Wout, dWout);
  vecAddInPlace(grads.bout, dLogits);

  // 3. Initialize dHidden for all positions (only last position gets gradient initially)
  const dHidden = new Array(len);
  for (let i = 0; i < len; i++) {
    dHidden[i] = zeros1D(embedDim);
  }
  vecAddInPlace(dHidden[len - 1], dLastHidden);

  // 4. Backward through blocks in reverse
  for (let blockIdx = numBlocks - 1; blockIdx >= 0; blockIdx--) {
    const block = params.blocks[blockIdx];
    const gBlock = grads.blocks[blockIdx];
    const bc = cache.blocks[blockIdx];

    // --- Second residual backward ---
    // hidden_out = hidden_pre_ffn + ffn2
    // dHidden flows to both branches
    const dFfn2 = new Array(len);
    for (let pos = 0; pos < len; pos++) {
      dFfn2[pos] = dHidden[pos].slice(); // copy, since residual keeps dHidden
    }

    // --- SwiGLU FFN backward ---
    // Forward was: gate = silu(W_gate @ normed2 + b_gate), up = W1 @ normed2 + b1
    //              gated = gate * up, ffn2 = W2 @ gated + b2
    for (let pos = 0; pos < len; pos++) {
      // W2 backward: ffn2 = matVec(W2, gated) + b2
      const { dMat: dW2, dVec: dGated } = matVecBackward(block.W2, bc.gated[pos], dFfn2[pos]);
      addMat(gBlock.W2, dW2);
      vecAddInPlace(gBlock.b2, dFfn2[pos]);

      // Product rule: gated = gate * up
      const dGate = vecMul(dGated, bc.up[pos]);
      const dUp = vecMul(dGated, bc.gate[pos]);

      // Gate backward: gate = silu(W_gate @ normed2 + b_gate)
      const dGatePre = siluBackward(bc.gatePre[pos], dGate);
      const { dMat: dW_gate, dVec: dNormed2_gate } = matVecBackward(block.W_gate, bc.normed2[pos], dGatePre);
      addMat(gBlock.W_gate, dW_gate);
      vecAddInPlace(gBlock.b_gate, dGatePre);

      // Up backward: up = W1 @ normed2 + b1 (no activation)
      const { dMat: dW1, dVec: dNormed2_up } = matVecBackward(block.W1, bc.normed2[pos], dUp);
      addMat(gBlock.W1, dW1);
      vecAddInPlace(gBlock.b1, dUp);

      // Combine both paths to dNormed2
      const dNormed2 = zeros1D(embedDim);
      vecAddInPlace(dNormed2, dNormed2_gate);
      vecAddInPlace(dNormed2, dNormed2_up);

      // --- RMSNorm2 backward ---
      const { dX: dPreNorm2, dGamma: dLn2G } = rmsNormBackward(
        bc.preNorm2[pos], block.ln2_g, dNormed2
      );
      vecAddInPlace(gBlock.ln2_g, dLn2G);

      // Add to residual path (dHidden already has the residual gradient)
      vecAddInPlace(dHidden[pos], dPreNorm2);
    }

    // --- First residual backward ---
    // hidden_post_attn = hidden_pre_attn + attnProj
    // dHidden flows to both branches
    const dAttnProj = new Array(len);
    for (let pos = 0; pos < len; pos++) {
      dAttnProj[pos] = dHidden[pos].slice();
    }

    // --- Wo backward: attnProj[pos] = matVec(Wo, attnOutputs[pos]) + bo ---
    const dAttnOutputs = new Array(len);
    for (let pos = 0; pos < len; pos++) {
      dAttnOutputs[pos] = zeros1D(embedDim);
    }
    for (let pos = 0; pos < len; pos++) {
      const { dMat: dWo, dVec: dAttnOut } = matVecBackward(block.Wo, bc.attnOutputs[pos], dAttnProj[pos]);
      addMat(gBlock.Wo, dWo);
      vecAddInPlace(gBlock.bo, dAttnProj[pos]);
      vecAddInPlace(dAttnOutputs[pos], dAttnOut);
    }

    // --- Attention backward ---
    const dQ = new Array(len);
    const dK = new Array(len);
    const dV = new Array(len);
    for (let pos = 0; pos < len; pos++) {
      dQ[pos] = zeros1D(embedDim);
      dK[pos] = zeros1D(embedDim);
      dV[pos] = zeros1D(embedDim);
    }

    for (let head = 0; head < numHeads; head++) {
      const offset = head * headDim;

      for (let i = 0; i < len; i++) {
        // Backward through weighted value sum:
        // attnOutputs[i][offset+d] = sum_j(weights[j] * V[j][offset+d])
        const dWeights = new Float32Array(len);
        for (let d = 0; d < headDim; d++) {
          const dOut = dAttnOutputs[i][offset + d];
          for (let j = 0; j < len; j++) {
            dWeights[j] += dOut * bc.V[j][offset + d];
            dV[j][offset + d] += dOut * bc.attnWeights[head][i][j];
          }
        }

        // Backward through softmax
        const dScores = softmaxBackward(bc.attnWeights[head][i], dWeights);

        // Backward through scaling and causal mask
        const scale = 1 / Math.sqrt(headDim);
        for (let j = 0; j <= i; j++) {
          const dScoreScaled = dScores[j] * scale;
          // score = Q[i] · K[j] (at head dimensions)
          for (let d = 0; d < headDim; d++) {
            dQ[i][offset + d] += dScoreScaled * bc.K[j][offset + d];
            dK[j][offset + d] += dScoreScaled * bc.Q[i][offset + d];
          }
        }
        // Masked positions (j > i) had score = -1e9, softmax grad is ~0, no gradient to propagate
      }
    }

    // --- RoPE backward: un-rotate dQ and dK ---
    for (let head = 0; head < numHeads; head++) {
      const off = head * headDim;
      for (let pos = 0; pos < len; pos++) {
        const dqSlice = dQ[pos].slice(off, off + headDim);
        const dkSlice = dK[pos].slice(off, off + headDim);
        dQ[pos].set(applyRoPEBackward(dqSlice, pos, freqs), off);
        dK[pos].set(applyRoPEBackward(dkSlice, pos, freqs), off);
      }
    }

    // --- Q, K, V projection backward ---
    const dNormed1 = new Array(len);
    for (let pos = 0; pos < len; pos++) {
      dNormed1[pos] = zeros1D(embedDim);
    }

    for (let pos = 0; pos < len; pos++) {
      // Q = matVec(Wq, normed1) + bq
      const { dMat: dWq, dVec: dN1_q } = matVecBackward(block.Wq, bc.normed1[pos], dQ[pos]);
      addMat(gBlock.Wq, dWq);
      vecAddInPlace(gBlock.bq, dQ[pos]);
      vecAddInPlace(dNormed1[pos], dN1_q);

      // K = matVec(Wk, normed1) + bk
      const { dMat: dWk, dVec: dN1_k } = matVecBackward(block.Wk, bc.normed1[pos], dK[pos]);
      addMat(gBlock.Wk, dWk);
      vecAddInPlace(gBlock.bk, dK[pos]);
      vecAddInPlace(dNormed1[pos], dN1_k);

      // V = matVec(Wv, normed1) + bv
      const { dMat: dWv, dVec: dN1_v } = matVecBackward(block.Wv, bc.normed1[pos], dV[pos]);
      addMat(gBlock.Wv, dWv);
      vecAddInPlace(gBlock.bv, dV[pos]);
      vecAddInPlace(dNormed1[pos], dN1_v);
    }

    // --- RMSNorm1 backward ---
    for (let pos = 0; pos < len; pos++) {
      const { dX: dPreNorm1, dGamma: dLn1G } = rmsNormBackward(
        bc.preNorm1[pos], block.ln1_g, dNormed1[pos]
      );
      vecAddInPlace(gBlock.ln1_g, dLn1G);

      // Add RMSNorm1 grad to residual path
      vecAddInPlace(dHidden[pos], dPreNorm1);
    }
  }

  // 5. Embedding backward: hidden[pos] = embedding[token] (no posEmbedding — RoPE)
  for (let pos = 0; pos < len; pos++) {
    const token = cache.tokens[pos];
    vecAddInPlace(grads.embedding[token], dHidden[pos]);
  }
}

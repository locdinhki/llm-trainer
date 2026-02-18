import { softmax, vecAddInPlace } from "./math.js";

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

export function layerNormBackward(x, g, dOut) {
  const n = x.length;

  // Recompute forward stats
  let sum = 0, sum2 = 0;
  for (let i = 0; i < n; i++) {
    sum += x[i];
    sum2 += x[i] * x[i];
  }
  const mean = sum / n;
  const variance = sum2 / n - mean * mean;
  const invStd = 1 / Math.sqrt(variance + 1e-5);

  const xHat = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    xHat[i] = (x[i] - mean) * invStd;
  }

  // Gradients for gamma and beta
  const dGamma = new Float32Array(n);
  const dBeta = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    dGamma[i] = dOut[i] * xHat[i];
    dBeta[i] = dOut[i];
  }

  // Gradient for input x
  // dxhat[i] = dOut[i] * g[i]
  // dx[i] = invStd * (dxhat[i] - mean(dxhat) - xHat[i] * mean(dxhat * xHat))
  let meanDxhat = 0;
  let meanDxhatXhat = 0;
  for (let i = 0; i < n; i++) {
    const dxh = dOut[i] * g[i];
    meanDxhat += dxh;
    meanDxhatXhat += dxh * xHat[i];
  }
  meanDxhat /= n;
  meanDxhatXhat /= n;

  const dX = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    dX[i] = invStd * (dOut[i] * g[i] - meanDxhat - xHat[i] * meanDxhatXhat);
  }

  return { dX, dGamma, dBeta };
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
    posEmbedding: zeros2D(seqLen, embedDim),
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
      ln1_b: zeros1D(embedDim),
      W1: zeros2D(ffnDim, embedDim),
      b1: zeros1D(ffnDim),
      W2: zeros2D(embedDim, ffnDim),
      b2: zeros1D(embedDim),
      ln2_g: zeros1D(embedDim),
      ln2_b: zeros1D(embedDim),
    });
  }
  return grads;
}

// --- Main backward function ---

export function backward(params, cache, target, config, grads) {
  const { embedDim, numHeads, numBlocks = 1 } = config;
  const headDim = embedDim / numHeads;
  const len = cache.tokens.length;

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

    // --- FFN2 backward: ffn2[pos] = matVec(W2, ffn1[pos]) + b2 ---
    for (let pos = 0; pos < len; pos++) {
      const { dMat: dW2, dVec: dFfn1 } = matVecBackward(block.W2, bc.ffn1[pos], dFfn2[pos]);
      addMat(gBlock.W2, dW2);
      vecAddInPlace(gBlock.b2, dFfn2[pos]);

      // --- ReLU backward ---
      const dFfn1Pre = reluBackward(bc.ffn1Pre[pos], dFfn1);

      // --- FFN1 backward: ffn1Pre[pos] = matVec(W1, normed2[pos]) + b1 ---
      const { dMat: dW1, dVec: dNormed2 } = matVecBackward(block.W1, bc.normed2[pos], dFfn1Pre);
      addMat(gBlock.W1, dW1);
      vecAddInPlace(gBlock.b1, dFfn1Pre);

      // --- LN2 backward ---
      const { dX: dPreNorm2, dGamma: dLn2G, dBeta: dLn2B } = layerNormBackward(
        bc.preNorm2[pos], block.ln2_g, dNormed2
      );
      vecAddInPlace(gBlock.ln2_g, dLn2G);
      vecAddInPlace(gBlock.ln2_b, dLn2B);

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

    // --- LN1 backward ---
    for (let pos = 0; pos < len; pos++) {
      const { dX: dPreNorm1, dGamma: dLn1G, dBeta: dLn1B } = layerNormBackward(
        bc.preNorm1[pos], block.ln1_g, dNormed1[pos]
      );
      vecAddInPlace(gBlock.ln1_g, dLn1G);
      vecAddInPlace(gBlock.ln1_b, dLn1B);

      // Add LN1 grad to residual path
      vecAddInPlace(dHidden[pos], dPreNorm1);
    }
  }

  // 5. Embedding backward: hidden[pos] = embedding[token] + posEmbedding[pos]
  for (let pos = 0; pos < len; pos++) {
    const token = cache.tokens[pos];
    vecAddInPlace(grads.embedding[token], dHidden[pos]);
    vecAddInPlace(grads.posEmbedding[pos], dHidden[pos]);
  }
}

// Checkpoint save/load with Float32Array <-> base64 serialization.

function float32ToBase64(f32arr) {
  const bytes = new Uint8Array(f32arr.buffer, f32arr.byteOffset, f32arr.byteLength);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function base64ToFloat32(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Float32Array(bytes.buffer);
}

function serializeParams(params) {
  return {
    embedding: params.embedding.map(float32ToBase64),
    Wout: params.Wout.map(float32ToBase64),
    bout: float32ToBase64(params.bout),
    blocks: params.blocks.map((block) => {
      const serialized = {};
      for (const key of Object.keys(block)) {
        const val = block[key];
        if (Array.isArray(val)) {
          serialized[key] = val.map(float32ToBase64);
        } else {
          serialized[key] = float32ToBase64(val);
        }
      }
      return serialized;
    }),
  };
}

function deserializeParams(data) {
  return {
    embedding: data.embedding.map(base64ToFloat32),
    Wout: data.Wout.map(base64ToFloat32),
    bout: base64ToFloat32(data.bout),
    blocks: data.blocks.map((block) => {
      const deserialized = {};
      for (const key of Object.keys(block)) {
        const val = block[key];
        if (Array.isArray(val)) {
          deserialized[key] = val.map(base64ToFloat32);
        } else {
          deserialized[key] = base64ToFloat32(val);
        }
      }
      return deserialized;
    }),
  };
}

export function saveCheckpoint(params, adamState, step, config, lossHistory) {
  return JSON.stringify({
    version: 1,
    config,
    step,
    lossHistory: lossHistory.slice(-200),
    params: serializeParams(params),
    adamState: {
      m: serializeParams(adamState.m),
      v: serializeParams(adamState.v),
      t: adamState.t,
    },
  });
}

export function loadCheckpoint(jsonString) {
  const data = JSON.parse(jsonString);
  if (data.version !== 1) throw new Error("Unknown checkpoint version");
  return {
    config: data.config,
    step: data.step,
    lossHistory: data.lossHistory,
    params: deserializeParams(data.params),
    adamState: {
      m: deserializeParams(data.adamState.m),
      v: deserializeParams(data.adamState.v),
      t: data.adamState.t,
    },
  };
}

const ARRAY_STRIDES = {
  centers: 3,
  scales: 3,
  quaternions: 4,
  colors: 3,
  color_gradients: 9,
  weights: 1,
  time_sigmas: 1,
  slice_velocities: 2,
  slice_roots: 4,
  importance: 1,
};

export async function loadJewelField(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not load Jewel field (${response.status}).`);
  }
  const payload = await response.json();
  if (payload.schema !== "spacetime-jewel-viewer-v1") {
    throw new Error(`Unsupported Jewel field schema: ${payload.schema ?? "missing"}.`);
  }
  const count = Number(payload.field?.count);
  if (!Number.isInteger(count) || count <= 0) {
    throw new Error("The Jewel field has no valid primitive count.");
  }
  if (!Array.isArray(payload.field.shape) || payload.field.shape.length !== 3) {
    throw new Error("The Jewel field is missing its (frames, height, width) shape.");
  }

  const arrays = {};
  for (const [name, stride] of Object.entries(ARRAY_STRIDES)) {
    const values = payload.arrays?.[name];
    if (!Array.isArray(values) || values.length !== count * stride) {
      throw new Error(`${name} must contain ${count * stride} numeric values.`);
    }
    arrays[name] = Float32Array.from(values);
  }
  return {
    ...payload,
    field: {
      ...payload.field,
      count,
      shape: payload.field.shape.map(Number),
      background: payload.field.background.map(Number),
      t_scale: Number(payload.field.t_scale),
    },
    arrays,
  };
}

# `field-loader.js`

## Purpose

Loads and validates the exported Jewel JSON before any GPU resources are created. Numeric arrays
are converted to typed buffers once so both Three.js render paths share the same source data.

## Components

### `loadJewelField`

- **Does**: Fetches the `spacetime-jewel-viewer-v1` payload, validates every array stride, and
  returns typed arrays plus normalized field metadata.
- **Interacts with**: `export_spacetime_viewer.py`, `main.js`, `slice-renderer.js`, and
  `volume-scene.js`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `main.js` | Rejected promise contains a useful user-facing error | Silent validation failures |
| Render modules | All arrays are `Float32Array` with declared strides | Array names or strides |

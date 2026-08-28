# `export_spacetime_viewer.py`

## Purpose

Exports a real fitted `PrimitiveField` checkpoint into the compact JSON schema consumed by the
interactive Three.js spacetime viewer. The export retains every Jewel and the parameters needed to
render fixed-time Gaussian cross-sections in the browser.

## Components

### `build_viewer_payload`

- **Does**: Validates checkpoint tensors and exports centers, principal scales, normalized
  quaternions, P1 color, opacity, and slice-derived geometry.
- **Interacts with**: `quat_to_rotmat` in `stprim/core/params.py` and `field-loader.js` in the viewer.
- **Rationale**: Conditional slice velocity and covariance are derived once in PyTorch so the web
  renderer receives stable, directly testable values.

### `main`

- **Does**: Loads a checkpoint, records its SHA-256 provenance, and writes minified JSON.
- **Interacts with**: `public/data/singer-field.json`, generated from the fitted singer checkpoint.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `field-loader.js` | `spacetime-jewel-viewer-v1` with flat numeric arrays | Schema or array stride |
| `slice-renderer.js` | Slice roots, velocities, time sigmas, P1 gradients, and weights | Removing derived arrays |
| `volume-scene.js` | Centers, scales, quaternions, colors, and importance | Coordinate or quaternion order |
| Reproduction workflow | Source filename and SHA-256 in the payload | Removing provenance |

## Notes

- The checkpoint stores quaternions as `w,x,y,z`; the viewer payload uses Three.js order
  `x,y,z,w`.
- The browser slice is an interactive positive-additive preview. Quantitative evaluation continues
  to use the PyTorch support-complete renderer.

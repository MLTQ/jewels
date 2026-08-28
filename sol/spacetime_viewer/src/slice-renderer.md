# `slice-renderer.js`

## Purpose

Renders one fixed-time cross-section of every fitted Jewel into an offscreen texture. The texture
is mapped directly onto the moving plane in the 3D volume, so the playhead is visibly both a slice
and a video frame.

## Components

### `SliceRenderer`

- **Does**: Builds one instanced quad per Jewel and evaluates its conditional 2D Gaussian, temporal
  attenuation, opacity, and P1 local color in a shader.
- **Interacts with**: Typed arrays from `field-loader.js`; its render-target texture is consumed by
  `volume-scene.js`.
- **Rationale**: Rendering the slice from the field avoids substituting a pre-rendered video texture
  for the operation the demo is meant to explain.

### `render`

- **Does**: Clears to the fitted background and additively accumulates all active Jewel
  contributions at the requested normalized time.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `volume-scene.js` | Stable `target.texture` with video aspect ratio | Target format or orientation |
| `main.js` | `render(renderer, time)` updates before the 3D pass | Method signature |
| Viewer disclosure | Interactive preview clamps each Jewel's negative RGB contribution | Claiming metric-renderer parity |

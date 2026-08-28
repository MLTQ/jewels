# `volume-scene.js`

## Purpose

Builds the orbitable Three.js view of the fitted `(u,v,t)` volume and places the live frame texture
on a plane that moves only along the time axis.

## Components

### `VolumeScene`

- **Does**: Owns the WebGL renderer, camera, orbit controls, field bounds, centroid cloud,
  covariance shells, and playhead plane.
- **Interacts with**: Field buffers from `field-loader.js` and the texture from
  `slice-renderer.js`.

### Centroid cloud

- **Does**: Displays every fitted center and brightens Jewels near the current time slice.
- **Rationale**: All 6,471 positions remain visible without turning thousands of overlapping
  translucent shells into an opaque block.

### Covariance shells

- **Does**: Displays the 500 highest-importance two-sigma ellipsoid glyphs using each Jewel's actual
  scale and rotation. A single blue outline keeps overlapping geometry legible; field colors remain
  visible on every centroid and in the live slice.
- **Rationale**: The ranked subset exposes anisotropy and tilt while the live slice still renders
  all Jewels.

### Playhead plane

- **Does**: Moves along `t`, carries the current rendered frame, and uses a teal border to remain
  visible through the field.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `main.js` | `renderer`, `setTime`, `resize`, and `render` | Public method names |
| Coordinate explanation | `u` is horizontal, `v` is screen-down, and `t` is depth | Axis remapping |
| Slice renderer | Plane texture has the source video's aspect and orientation | Texture orientation |

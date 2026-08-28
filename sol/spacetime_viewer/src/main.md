# `main.js`

## Purpose

Coordinates data loading, playback state, the offscreen Gaussian slice, the 3D scene, resizing, and
accessible transport updates.

## Components

### `setFrame`

- **Does**: Converts an integer source frame to normalized time, renders that cross-section, moves
  the playhead, and updates the readout.
- **Interacts with**: `SliceRenderer.render` and `VolumeScene.setTime`.

### `setPlaying`

- **Does**: Toggles the 12-fps play loop and keeps the button's visible and ARIA state aligned.

### `start`

- **Does**: Loads the fitted field, constructs both render paths, binds native controls, and starts
  the animation loop.
- **Interacts with**: `field-loader.js`, `slice-renderer.js`, and `volume-scene.js`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `index.html` | Element IDs used during startup | Renaming IDs |
| Viewer operator | `/data/singer-field.json` exists after export | Data path changes |
| Playback explanation | One frame maps linearly to one fixed `t` slice | Nonlinear time mapping |

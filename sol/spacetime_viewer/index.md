# `index.html`

## Purpose

Defines the standalone spacetime viewer's accessible interface: one dominant 3D canvas, a small
legend, and a play/scrub transport. The document declares UTF-8 explicitly so mathematical axis
labels render consistently. The frame plane itself carries the reconstructed video image.

## Components

### `#volume-canvas`

- **Does**: Hosts the orbitable Three.js field and moving cross-section.
- **Interacts with**: `main.js` and `volume-scene.js`.

### Playback controls

- **Does**: Expose play/pause, direct frame scrubbing, and the current frame count.
- **Interacts with**: `main.js`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `main.js` | Stable element IDs for canvas, status, button, slider, and readout | Renaming IDs |
| Keyboard and assistive technology users | Native button/range controls and live status | Replacing semantic controls |

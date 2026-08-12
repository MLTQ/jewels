# `test_multiscale_video_guide.py`

## Purpose

Protects spatial/temporal alignment and within-cell information in the multiscale semantic-guide
boundary.

## Components

### `MultiscaleVideoGuideTests`

- **Does**: Verifies fine tokens retain distinct subcell RGB samples, canonical cells flatten as
  `(u,v,t)`, and the default pyramid exposes finite temporal derivative channels.
- **Interacts with**: `multiscale_video_guide.py` and `GridSpec`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Video-to-jewel realizer | Default guide shape is `(cells,24,16)` | Scale/subgrid defaults |
| Cell/rank attention | Token positions correspond to the addressed jewel cell | Axis ordering |

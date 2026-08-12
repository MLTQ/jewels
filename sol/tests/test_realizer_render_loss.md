# `test_realizer_render_loss.py`

## Purpose

Protects the denoised-endpoint and differentiable visual-supervision contracts of the multiscale
video-to-jewel realizer.

## Components

### `RealizerRenderLossTests`

- **Does**: Verifies exact flow velocity reconstructs the clean target, visual terms remain finite
  and backpropagate into predicted jewels, and identical fields produce zero loss.
- **Interacts with**: `realizer_render_loss.py`, the exact renderer, and canonical synthetic jewels.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Realizer trainer | Render loss differentiates through predicted 22-D marks | Gradient path |
| Loss calibration | Identical fixed-topology fields score zero | Objective baseline |

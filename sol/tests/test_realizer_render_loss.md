# `test_realizer_render_loss.py`

## Purpose

Protects the denoised-endpoint and differentiable visual-supervision contracts of the multiscale
video-to-jewel realizer.

## Components

### `RealizerRenderLossTests`

- **Does**: Verifies exact flow velocity reconstructs the clean target, visual terms remain finite
  and backpropagate into predicted jewels, identical fields produce zero loss, and an anchored
  patch includes local frontier time zero. A candidate-only background must also receive a finite
  visual gradient against the fixed target background.
- **Saliency checks**: A one-hot canonical cell maps to its intended render patch, scaffold motion
  and rare chroma outrank static background, and foreground/motion/stability terms remain finite
  with finite mark gradients.
- **Interacts with**: `realizer_render_loss.py`, the exact renderer, and canonical synthetic jewels.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Realizer trainer | Render loss differentiates through predicted 22-D marks | Gradient path |
| Loss calibration | Identical fixed-topology fields score zero | Objective baseline |
| Boundary fine-tune | Anchored sampling starts the first patch at the frontier | Patch indexing |
| Motion-aware fine-tune | Canonical guide cells address their matching render regions | Grid order |
| Background memorization | Candidate RGB is differentiable without changing the target render | Composition policy |

# `train_amortized_encoder.py`

## Purpose

Trains the amortized encoder with exactly the fitter's proven objective — stochastic-voxel MSE
against the source video — amortized across windows: each step encodes one full training window
and renders a fresh random voxel batch differentiably through the Cholesky path.

For scaling gates it trains in full shuffled corpus passes and stops only after a frozen
validation score has failed to improve by the declared PSNR margin for a declared number of
evaluations. This replaces fixed-step comparisons where smaller corpora received many more
exposures per clip.

## Components

### `sample_voxels`
- **Does**: Draws random `(t,y,x)` voxel centers, maps them to normalized `(u,v,t)` with the
  exact `frame_times`/linspace conventions the fitted fields use, and returns their RGB values.

### `main`
- **Does**: Loads every manifest window at fit geometry, trains on the train split, evaluates
  held-out sampled-voxel PSNR per validation window at a fixed eval seed, and checkpoints with
  manifest digest and full provenance.
- **Rationale**: No feature-space loss and no fitted-field supervision in v0 — the gate asks
  what render supervision alone achieves; fitted targets remain available for aux losses if the
  gate demands them.
- **Does**: `--max-epochs` traverses each nested subset once per epoch in a deterministic seeded
  shuffle, evaluates at matched epoch intervals, saves the best checkpoint separately from the
  latest recovery checkpoint, applies warmup in corpus passes rather than raw steps, and applies
  a declared validation-plateau stopping rule. The best checkpoint tracks every strict score
  improvement; the larger minimum-delta threshold is used only to reset plateau patience.
- **Does**: `--resume` validates architecture and manifest ownership, restores model and optimizer,
  and keeps global step/epoch provenance while applying a fresh, explicitly declared continuation
  schedule. This supports a common low-rate convergence phase without rewriting first-stage data.
- **Does**: Defaults to the five-sigma support-complete tiled renderer; the all-center renderer
  remains available as an oracle for small correctness checks.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| jewels-sb0 gate | Held-out evaluation at fixed seed comparable across runs | Eval protocol |
| Encode/audit CLI | Checkpoint meta carries grid, slots, and manifest digest | Meta schema |
| Scaling reports | `encoder.pt` is best validation checkpoint; `latest.pt` is last recovery state | Checkpoint naming |

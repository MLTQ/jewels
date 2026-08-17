# `train_amortized_encoder.py`

## Purpose

Trains the amortized encoder with exactly the fitter's proven objective — stochastic-voxel MSE
against the source video — amortized across windows: each step encodes one full training window
and renders a fresh random voxel batch differentiably through the Cholesky path.

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

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| jewels-sb0 gate | Held-out evaluation at fixed seed comparable across runs | Eval protocol |
| Encode/audit CLI | Checkpoint meta carries grid, slots, and manifest digest | Meta schema |

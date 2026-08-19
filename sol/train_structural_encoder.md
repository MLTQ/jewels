# `train_structural_encoder.py`

## Purpose

Trains the Path B encoder with the fitter's own objective — stochastic-voxel render MSE — and
reports **structure alongside quality** at every evaluation, because the experiment's question
is not only "does it reconstruct" but "does it produce tubes clustered on content".

## Components

### `main`
- **Does**: Trains on the five-domain manifest, evaluating held-out sampled-voxel PSNR plus
  anisotropy, extent variation, and occupancy uniformity via `structure_report`.
- **Rationale**: No explicit anisotropy prior is applied. If tubes emerge from scarcity and
  expressiveness alone that is the stronger result; if they do not, the need for a structural
  prior is itself the finding.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Path B gate | Evaluation carries `structure` block beside `macro_psnr` | Report schema |

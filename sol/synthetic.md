# `synthetic.py`

## Purpose

Creates deterministic canonical jewel tensors for fast CPU research checks. Synthetic fixtures keep
the spike runnable without private corpora or external checkpoints.

## Components

### `random_jewels`
- **Does**: Produces valid `(N,22)` features with uniform centers, isotropic log covariance, color,
  and opacity.
- **Interacts with**: Token-grid, edit, renderer, and autoencoder tests.

### `elongated_knn_counterexample`
- **Does**: Constructs 64 close zero-color jewels that hide one farther broad contributor from a
  center-kNN renderer.
- **Rationale**: Encodes the exact failure mode that a motion-aligned elongated jewel can trigger.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `spike.py` and tests | Canonical feature layout and deterministic seeds | Feature layout or RNG policy |

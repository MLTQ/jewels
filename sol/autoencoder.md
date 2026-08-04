# `autoencoder.py`

## Purpose

Defines the learned count-aware bottleneck between fitted jewel sets and raster latents. Encoding is
linear in jewel count; deterministic decoding gives all sampling responsibility to the latent prior.

## Components

### `OccupancyAwareEncoder`
- **Does**: Pools projected mean, variance, count, and occupancy before cell-level attention.
- **Local-only mode**: `depth=0` omits quadratic global attention while retaining the same raster
  token contract, which makes fine spacetime grids practical for fidelity experiments.
- **Interacts with**: Cell indexing and capacity from `token_grid.py`.

### `StructuredSlotDecoder`
- **Does**: Expands each raster latent into canonical jewel slots, existence logits, and a count.
- **Rationale**: Shared-latent residual MLP slots are linear in slot count; deterministic outputs remain
  coordinated through the cell latent without quadratic slot attention.

### `StructuredJewelAutoencoder`
- **Does**: Supplies masked feature/existence/count losses and variable-size decoding.
- **Interacts with**: `RasterFlowPrior` models its encoder output.

### `StructuredJewelAutoencoder.loss_from_compact`
- **Does**: Trains from precomputed occupied cell/slot targets without dense empty target tensors.

### `StructuredJewelAutoencoder.structural_loss`
- **Does**: Scores a previously decoded slot field so the corpus trainer can add differentiable
  render loss without running the model twice.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `latent_prior.py` | Encoder returns `(B,n_cells,latent_dim)` in raster order | Latent layout |
| Future training CLI | Structural loss reports feature, existence, and count components | Loss schema |
| Future checkpoints | Grid spec and all model dimensions are serialized | Architecture defaults |

## Notes

- Render/perceptual losses are composed in the corpus training loop; equal feature loss is
  insufficient.
- Canonical ranks can swap near ties. Compare with per-cell Hungarian or optimal transport on real
  fits before committing to a long run.
- Decode clamps predicted log-counts to the declared cell capacity before exponentiation.
- Decoded centers are constrained by construction to the cell represented by their latent.
- Existence BCE balances occupied slots against the usually much larger empty-slot population.
- Cell sums and second moments accumulate in fp32 even under fp16 AMP.
- Encoder depth must be nonnegative; zero intentionally selects the local-only codec path.

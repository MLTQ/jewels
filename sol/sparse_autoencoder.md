# `sparse_autoencoder.py`

## Purpose

Defines the dense-corpus tokenizer without worst-cell padding. It preserves the editable raster
latent while decoding exactly the requested canonical jewel ranks in each cell.

## Components

### `SparseSlotDecoder`
- **Does**: Maps `(cell latent, cell ID, continuous rank embedding)` tuples to canonical jewel
features and predicts one count per cell.
- **Rationale**: The dense corpus reaches 399 jewels in one 12×12×6 cell. Padding all 864 cells to
  that occupancy would evaluate over 344k slots to reconstruct 45k jewels.

### `SparseSlotDecoder.decode_indices`
- **Does**: Evaluates only occupied/requested ranks in bounded-memory chunks.

### `SparseSlotDecoder.decode`
- **Does**: Converts predicted log-counts to variable-size per-example jewel tensors with no
  existence-padding field.

### `SparseJewelAutoencoder`
- **Does**: Reuses the count-aware raster encoder and supplies compact feature/count training losses.
- **Interacts with**: `token_grid.py`, `autoencoder.OccupancyAwareEncoder`, and the raster prior.

### `RankConditionedEncoder`
- **Does**: Applies a nonlinear projection to each `(jewel feature, canonical rank)` pair before
  accumulating per-cell moments.
- **Rationale**: Plain permutation-invariant moments describe a local distribution but do not bind
  individual detail to the ranks requested by the deterministic decoder. Rank conditioning retains
  that correspondence without changing the one-token-per-cell editing contract.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Dense trainer | Output contains `(B,N,F)` occupied features and `(B,C)` log-count | Output schema |
| Evaluation/prior | `encoder` returns raster latents and `decode` returns variable sets | Codec interface |
| Editor | Cell-constrained centers and canonical rank semantics remain stable | Spatial/rank layout |

## Notes

- Count prediction replaces padded existence logits; occupied ranks are contiguous by construction.
- Integer-rank Fourier features use 4/16/64-jewel wavelengths plus normalized/log rank. Typical
  22-jewel cells therefore span multiple phases instead of collapsing into the first few percent of
  a capacity-normalized low-frequency basis.
- Canonical rank swaps near center ties remain a research limitation shared with the padded decoder.
- `encoder_mode="pooled"` loads the earlier checkpoints; `"rank"` selects the correspondence-aware
  experiment. Both modes accept unordered input sets, because ranks are assigned canonically.

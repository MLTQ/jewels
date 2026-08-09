# `grouped_sparse_autoencoder.py`

## Purpose

Defines the occupied-group tokenizer experiment. It replaces one lossy moment token per raster cell
with a compact stream of local tokens, each responsible for at most a small canonical jewel group.

## Components

### `GroupedLatents`
- **Does**: Carries compact learned token values plus discrete batch, cell, group, and group-count
  topology.
- **Rationale**: Sparse addresses and token lengths are part of the editable/generative sequence;
  empty raster cells must not consume content vectors.

### `GroupedTokenEncoder`
- **Does**: Canonically partitions each occupied cell into fixed-maximum-size jewel groups and pools
  nonlinear feature/rank statistics independently for every group.
- **Interacts with**: `CompactGrid`, shared Fourier cell features, and the grouped decoder.
- **Rationale**: Small groups prevent unrelated colors and surfaces in the same cell from collapsing
  into one moment-like summary.

### `GroupedTokenDecoder`
- **Does**: Expands each occupied token into its recorded number of within-group ranks and constrains
  decoded centers to the token's parent cell.
- **Rationale**: Topology is lossless in the tokenizer round-trip; a future prior learns addresses,
  group lengths, and content rather than relying on a dense empty-cell count head.

### `GroupedSparseJewelAutoencoder`
- **Does**: Exposes the existing `encoder`, `decode`, `forward_compact`, and `structural_loss`
  interfaces while using ragged occupied tokens internally.
- **Interacts with**: `train_dense_autoencoder.py`, held-out evaluation, and visual renderers.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Dense trainer | Output features retain canonical target order and exact target count | Group ordering |
| Evaluation/rendering | `decode(encoder(features))` returns one variable jewel tensor per batch | Latent schema |
| Future prompt prior | Token topology explicitly contains cell address, group index, and group length | Topology fields |
| Editor | Every decoded center remains inside its addressed spacetime cell | Center constraint |

## Notes

- This is a representation proof, not yet the final generative hierarchy. The prompt prior must
  generate the discrete sparse topology as well as continuous token content.
- Canonical rank ties inherit the current token-grid limitation; learned or optimal-transport
  grouping remains future work.

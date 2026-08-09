# `test_sparse_autoencoder.py`

## Purpose

Protects the dense tokenizer's central scaling contract: training and inference allocate requested
jewels, not `cells × maximum_slots` padding.

## Components

### `SparseAutoencoderTests`
- **Does**: Verifies compact loss gradients, chunked occupied decoding, exact predicted counts, and
  cell-constrained output centers, distinct multi-scale canonical-rank embeddings, and the
  fine-grid local-only encoder path. It also proves that rank-conditioned encoding remains invariant
  to the input set's incidental order. Fourier-position coverage proves every raster cell has a
  distinct shared basis, removes absolute lookup tables, backpropagates, and rejects the unsupported
  legacy pooled-encoder combination. A sparse-grid loss control verifies occupancy balancing
  increases pressure on rare positive cells without changing the default objective.
- **Interacts with**: `sparse_autoencoder.py` and `token_grid.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Dense tokenizer research | Memory/output scale with actual jewel count | Decoder allocation semantics |

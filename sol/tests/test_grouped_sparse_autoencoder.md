# `test_grouped_sparse_autoencoder.py`

## Purpose

Protects the occupied-group tokenizer's sparse topology and reconstruction contracts.

## Components

### `GroupedSparseAutoencoderTests`
- **Does**: Verifies exact topology-derived jewel counts, parent-cell center constraints, compact
  token allocation, canonical permutation invariance, and differentiable feature reconstruction.
- **Interacts with**: `grouped_sparse_autoencoder.py` and `token_grid.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Grouped tokenizer experiment | Tokens scale with occupied groups and decode in canonical order | Topology/order semantics |
| Future sparse prior | Group lengths sum to the generated jewel count | Group-count representation |

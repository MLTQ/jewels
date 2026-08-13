# `test_train_scaffold_mark_flow.py`

## Purpose

Protects device-row preparation for the universal 1,024-rank scaffold mark trainer.

## Components

### `TrainScaffoldMarkFlowTests`

- **Does**: Verifies train-only rows include all initial views, frontier-zero context stays exactly
  zero, canonical ranks remain within capacity, and render-supervision metadata is complete.
- **Saliency checks**: Prepared rows own a positive cell-importance raster and equal mark errors in
  a high-saliency cell receive greater objective weight than errors in a quiet cell.
- **Lifecycle check**: Spatial-appearance mode leaves temporal-dimension errors uniformly weighted
  even when their rows occupy cells with very different saliency.
- **Background check**: Single-field initialization is the exact causal initial-guide mean and
  rejects multiple physical training sources.
- **Precision check**: Rendered supervision disables unsafe mixed-precision loss scaling while
  feature-only CUDA training retains it.
- **Interacts with**: `train_scaffold_mark_flow.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Training loop | Initial rows are not filtered as empty-prefix examples | Preparation policy |
| Mark model | Prepared cell/rank tensors honor the shared grid | Capacity/order |
| Render fine-tune | Every row carries background and timeline metadata | Preparation schema |
| Feature saliency | Every row carries one mean-normalized importance per guide cell | Weighting schema |
| Background overfit | Initialization never reads fitted background metadata | Gauge ownership |

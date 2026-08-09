# `evaluate_dense_autoencoder.py`

## Purpose

Runs the fixed exact-render audit across a saved sparse tokenizer without resuming training. It is
separate from the padded-checkpoint evaluator and restores the checkpoint's explicit architecture.

## Components

### `main`
- **Does**: Validates the tokenizer architecture ID, restores model/grid/train-only normalization and
  held-out sources, evaluates all requested windows, and emits reproducible JSON.
- **Cross-domain mode**: `--all-sources` deliberately evaluates every supplied corpus example while
  retaining the frozen checkpoint normalization and weights. The output labels this selection so it
  cannot be confused with the original source-held-out Avenue protocol.
- **Capacity diagnostic**: `--slots-override N` may only increase the checkpoint's per-cell slot
  contract. It changes rank/count support but not weights, grid, normalization, or latent width;
  output records both capacities. This is an explicitly non-native transfer diagnostic, not the
  checkpoint's original protocol.
- **Interacts with**: `sparse_autoencoder.py`, `corpus.py`, and `evaluation.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Dense research audit | Macro-source/window PSNR, count ratio, and per-window metrics | Output schema |
| Future dense prior | Selected tokenizer checkpoint is source-held-out and reproducible | Meta schema |
| Transfer experiments | New-domain corpora can be audited without pretending source IDs match | Selection label |
| Capacity audit | No cross-domain jewel is silently dropped; overrides cannot reduce capacity | Slot metadata |

## Notes

- This isolates tokenizer error against the sharper dense fitted field, not raw source-video error.
- Dense-cell count prediction and sparse occupied-group topology use the identical render protocol;
  architecture-specific reconstruction cannot change the metric.

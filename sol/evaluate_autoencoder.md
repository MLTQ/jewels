# `evaluate_autoencoder.py`

## Purpose

Re-runs the fixed held-out render protocol from a saved tokenizer checkpoint without resuming or
mutating training. This supports broader evaluation than the inexpensive measurements embedded in
the training loop.

## Components

### `main`
- **Does**: Restores architecture, grid, train-only normalization, and held-out source identities
  from checkpoint metadata; loads the corpus; evaluates source-balanced windows; emits JSON.
- **Interacts with**: `corpus.py`, `autoencoder.py`, and `evaluation.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Research audit | Training checkpoint metadata fully reconstructs the protocol | Checkpoint meta schema |
| Result artifacts | JSON records checkpoint step, sample budget, sources, and per-window metrics | Output schema |

## Notes

- `--max-examples` is an upper bound; evaluation uses every held-out window when the bound is larger.
- Prefer `macro_source_psnr` for model selection when source videos contribute unequal window counts.
- This still compares fitted-field renders, not source pixels. It isolates tokenizer error from
  fitter error.

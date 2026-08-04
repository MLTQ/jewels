# `compare_tokenizers.py`

## Purpose

Makes tokenizer scaling decisions visually rather than from sampled PSNR alone. It renders matched
held-out targets through two checkpoints under the same production renderer and temporal indices.

## Components

### `main`
- **Does**: Verifies a common source holdout, selects source-balanced windows, renders target plus
  both deterministic round-trips, and writes labeled GIFs and a manifest.
- **Interacts with**: `autoencoder.py`, `corpus.py`, and rendering helpers from
  `render_prior_samples.py`.

### `_load_tokenizer` / `_roundtrip`
- **Does**: Restore checkpoint-specific architecture/normalization and decode a fitted example.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Tokenizer model selection | Panel order is target / baseline / candidate | Panel semantics |
| Research audit | Both checkpoints share held-out source IDs | Split provenance |

## Notes

- The labels describe the current 5.15× and 2.57× bottlenecks; change them if checkpoints with
  different compression are compared.

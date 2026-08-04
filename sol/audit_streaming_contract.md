# `audit_streaming_contract.py`

## Purpose

Runs the persistent streaming contract against one real fitted checkpoint and writes a reproducible
JSON report.

## Components

### `main`

- **Does**: loads canonical jewels, measures physical-time density/birth/lifespan statistics, builds
  rolling windows, and samples a monolithic-versus-carry/commit render comparison
- **Interacts with**: `streaming_metrics.py` and production `prior/featurize.py`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Result reports | Checkpoint contains `state` and `info.shape` | Checkpoint schema |
| Streaming gate | Render coverage has no missing/duplicate points and negligible error | Audit semantics |

## Notes

- `--fps` supplies physical seconds because historical checkpoints do not record source frame rate.
- The reference renderer is intentionally slow and finite-support; a few sampled points per frame
  are sufficient to validate subset equivalence.

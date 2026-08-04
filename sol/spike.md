# `spike.py`

## Purpose

Runs the central spike contracts on CPU without training or private data. Its JSON output is a quick
sanity record, not a quality benchmark.

## Components

### `main`
- **Does**: Losslessly packs 45k jewels, demonstrates center-kNN omission, builds a cursor translation
  plan, and checks exact clean-cell clamping during latent inpainting.
- **Interacts with**: Every core `sol/` module.

### `_toy_velocity`
- **Does**: Supplies a deterministic conditional velocity for exercising the sampler interface.
- **Rationale**: Tests mechanics only; it is not a learned video prior.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Developer workflow | `python -m sol.spike` emits JSON and exits successfully on CPU | CLI/module entry point |

## Notes

- Visual quality proof begins only when a corpus training CLI replaces `_toy_velocity` with a trained
  latent prior and evaluates held-out rendered reconstructions.

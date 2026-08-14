# `calibrate_coupled_set_checkpoint.py`

## Purpose

Creates a provenance-bearing inference calibration of a trained coupled-set checkpoint. It supports
a bounded strength screen when full coupling improves pixel fidelity but overshoots structural
similarity.

## Components

### `calibrated_checkpoint`

- **Does**: Scales only the zero-origin final residual projection in every `set_blocks.*` module.
- **Rationale**: Base weights remain frozen and exact; strength zero restores the original base
  function and strength one preserves the trained candidate.
- **Safety**: Rejects non-coupled, previously calibrated, incomplete, and out-of-range inputs;
  removes optimizer/scaler state because a calibrated artifact is evaluation-only.

### `main`

- **Does**: Loads one checkpoint, applies the declared strength, and saves the calibrated artifact.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Coupled-set evaluator | Checkpoint constructor arguments remain unchanged | Metadata/model args |
| Scientific report | Source path, strength, and scaled state names are serialized | Provenance |
| Future training | Calibrated artifacts are not resumable optimizer checkpoints | Optimizer policy |

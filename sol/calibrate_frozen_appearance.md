# `calibrate_frozen_appearance.py`

## Purpose

Measures loss values and appearance-gradient scales for a frozen residual checkpoint before any
optimizer is constructed. This prevents full-frame, temporal, range, or residual weights from being
selected after looking at training outcomes.

## Components

### `gradient_l2_norm`

- **Does**: Uses `torch.autograd.grad` to measure a joint parameter-gradient norm without filling
  `.grad` fields or mutating model state.

### `main`

- **Does**: Loads one manifest-owned video and one residual factorized checkpoint, freezes geometry,
  renders identical sampled points and contiguous low-resolution frames, then records every named
  loss, gradient norm, range fraction, and residual energy in a JSON report.
- **Safety**: Rejects non-residual or non-factorized checkpoints and records zero geometry-trainable
  parameters, no optimizer construction, and no optimizer steps.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Frozen appearance protocol | Calibration precedes optimization and identifies its source | Report schema |
| Weight registration | Component norms share one prediction graph and parameter set | Gradient semantics |
| GPU screening | Grid dimensions and support capacity are explicit | CLI defaults |

# `test_build_local_adapter_progression.py`

## Purpose

Protects the pitch progression sheet's audit ordering, dimensions, and per-style outputs.

## Components

### `LocalAdapterProgressionTests`
- **Does**: Builds a synthetic labeled audit with two styles and two checkpoints.
- **Does**: Verifies the lattice column is dropped, individual style strips are written, and the
  headline sheet includes space for metric annotations.
- **Does**: Opens generated image artifacts with bounded resource ownership.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `build_local_adapter_progression.py` | Synthetic column order matches the exact audit | Sheet-order semantics |

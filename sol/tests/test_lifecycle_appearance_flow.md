# `test_lifecycle_appearance_flow.py`

## Purpose

Protects the matched stochastic control that freezes temporal jewel state while allowing a second
mark flow to change geometry and appearance.

## Components

### `LifecycleAppearanceFlowTests`

- **Does**: Confirms the base branch is bit-identical to ordinary Euler sampling for the same seed,
  candidate appearance changes survive, lifecycle dimensions remain exact, and incompatible grids
  fail before sampling. A static-detail mask proves every omitted coordinate remains exact, and a
  zero-strength row remains fully identical to its base.
- **Interacts with**: `lifecycle_appearance_flow.py` and `birth_mark_flow.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Two-stream render gate | Adding the candidate branch does not perturb the base RNG trajectory | Step order |
| Lifecycle audit | Candidate temporal coordinates are assigned, not approximately penalized | Dimension split |

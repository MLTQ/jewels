# `test_render.py`

## Purpose

Protects the renderer correctness gate: an elongated contributor must survive conservative culling,
and finite-support error must remain measurable against an all-jewel reference.

## Components

### `RenderTests`
- **Does**: Exercises bounded eigensolve equivalence, the known center-kNN counterexample, and a
  random five-sigma audit.
- **Interacts with**: `render.py` and `synthetic.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Research renderer work | Exact renderer remains the oracle; conservative renderer retains support | Approximation semantics |

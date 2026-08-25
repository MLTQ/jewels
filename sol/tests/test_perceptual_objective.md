# `test_perceptual_objective.py`

## Purpose

Protects perceptual-training gradient and validation semantics without downloading LPIPS weights.

## Components

### `PerceptualObjectiveTests`
- **Does**: Uses an injected differentiable distance to prove gradients reach rendered pixels and
  verifies mismatched video shapes are rejected.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `perceptual_objective.py` | Tests remain independent of the optional `lpips` package | Eager LPIPS import |

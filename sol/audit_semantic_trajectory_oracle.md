# `audit_semantic_trajectory_oracle.py`

## Purpose

Runs Gate 2a10, replacing arbitrary donor-disagreement tubes with training-only semantic paths and a
fixed-candidate density-balanced boundary.

## Components

### `main`

- **Does**: Reuses the exact Gate 2a9 source-disjoint rendering battery, fitting only the semantic
  path and balance rule from registered training fields.
- **Does**: Applies the stricter 5% count-adjustment threshold and preserves the wrong-object and
  pooled-null causal controls.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a10 protocol | Semantic path and radius candidate rules are frozen | Mask selection |
| Cross-gate comparison | Arms/render points/evaluation grid match Gate 2a9 | Audit parity |
| Scientific review | Template-backed target-program oracle is labeled explicitly | Claim ownership |

# `test_plot_frozen_appearance_evidence.py`

## Purpose

Protects the scientific label mapping and generated figure for frozen-appearance evidence.

## Components

### `FrozenAppearanceEvidenceTests`

- **Does**: Builds a minimal registered result tree, verifies compute and ablation extraction, and
  requires a non-empty rendered graph.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `plot_frozen_appearance_evidence.py` | Generic audit arms map to stable scientific labels | Mapping semantics |

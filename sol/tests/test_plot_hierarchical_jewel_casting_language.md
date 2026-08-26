# `test_plot_hierarchical_jewel_casting_language.py`

## Purpose

Protects evidence ordering and explicit fresh-result labeling in the hierarchical pitch graph.

## Components

### `HierarchicalPlotTests`
- **Does**: Verifies that the hierarchy is appended after its precursor controls with verdict and
  metrics intact.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Hierarchical evidence | Fresh arm is last and never replaces precursor data | Plot payload order |

# `test_render_scaffold_topology_realizer.py`

## Purpose

Protects equal-source aggregation in the autonomous topology plus frozen-realizer visual gate.

## Components

### `RenderScaffoldTopologyRealizerTests`

- **Does**: Confirms arithmetic macro averaging with floating-point tolerance and rejects report
  sections with mismatched keys.
- **Interacts with**: `render_scaffold_topology_realizer.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Research summary | Every held-out class contributes one equal-weight record | Aggregation policy |
| JSON output | Metric sections have identical key sets before averaging | Report schema |

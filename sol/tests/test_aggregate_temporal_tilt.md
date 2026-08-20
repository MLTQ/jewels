# `test_aggregate_temporal_tilt.py`

## Purpose

Protects the decision-grade multi-source temporal-tilt replication gate.

## Components

### `source_report(source)`
- Builds a compact three-seed v2 report fixture with matched budgets and positive causal effects.

### `AggregateTemporalTiltTests`
- Confirms three sources and nine paired runs pass every gate when the effect is consistent.
- Confirms mixing incompatible fitting protocols fails loudly.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `sol/aggregate_temporal_tilt.py` | Nine-pair gate and strict protocol matching | Gate or validation semantics |

# `test_scaffold_mark_data.py`

## Purpose

Protects the initial-compatible mark corpus, empty-prefix raster, and causal selection of context
and carry from a generated append-only jewel field.

## Components

### `ScaffoldMarkDataTests`

- **Does**: Verifies initial births are trained, normalization is finite, empty context is explicit,
  and generated-state window indices reproduce the shared streaming lifecycle contract.
- **Interacts with**: `scaffold_mark_data.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Universal mark trainer | Initial and continuation views share one corpus | View filtering |
| Frontier-zero sampler | Empty context becomes a finite 46-channel zero raster | Empty handling |
| Persistent rollout | Context and carry are selected by row without modifying features | ID/lifecycle policy |

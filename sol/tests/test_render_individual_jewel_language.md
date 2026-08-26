# `test_render_individual_jewel_language.py`

## Purpose

Protects non-cherry-picked record selection for the passing individual-Jewel contact sheet.

## Components

### `IndividualJewelRenderTests`

- **Does**: Verifies protocol order, lowest fitter-seed selection, and missing-source rejection.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 0f renderer | Exactly one fixed source record per validation video | Selection semantics |

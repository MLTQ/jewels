# `test_plot_jewel_casting_language.py`

## Purpose

Protects evidence extraction from lexical vocabulary sorting and accidental report-schema drift.

## Components

### `CastingLanguagePlotTests`
- **Does**: Verifies numeric ordering, verdict propagation, and unknown-schema rejection.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Evidence plotter | Vocabulary keys are numeric strings in the JSON report | Report vocabulary keys |

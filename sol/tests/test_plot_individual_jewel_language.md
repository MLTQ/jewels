# `test_plot_individual_jewel_language.py`

## Purpose

Protects the evidence loader that feeds the passing individual-Jewel language plot.

## Components

### `IndividualJewelPlotTests`

- **Does**: Verifies exact metric extraction, lock-fraction percentage conversion, report verdict,
  and schema rejection.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 0f plot | Report values are not relabeled or recomputed post hoc | Metric ownership |

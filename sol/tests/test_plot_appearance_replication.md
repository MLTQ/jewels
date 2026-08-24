# `test_plot_appearance_replication.py`

## Purpose

Protects seed ordering and metric extraction for the residual-control replication screen.

## Components

### `AppearanceReplicationPlotTests`
- **Does**: Verifies seed-ordered PSNR, occupancy, active-fraction, and mixed-tilt parsing.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Replication graph | Input summary order is preserved | Reordering rows |

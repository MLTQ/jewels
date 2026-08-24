# `plot_appearance_replication.py`

## Purpose

Shows the final three-seed residual-control sampled screen separately from exact seed-0 evidence.
This makes the repeatable near-20 dB fidelity and the seed-1/2 occupancy failures visible together.

## Components

### `load_replication`
- **Does**: Reads final sampled PSNR, occupancy uniformity, active fraction, and mixed spacetime tilt
  from summaries supplied in seed order.

### `main`
- **Does**: Requires exactly three seed summaries and draws the frozen `20 dB`, `0.985`, `0.70`, and
  `0.25` screen thresholds.
- **Does**: Reserves metric-scaled annotation headroom so near-identical seed values remain legible.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Appearance-contract report | Summary order is seed 0, 1, 2 | CLI order |
| Replication interpretation | Occupancy is lower-is-more-irregular; other structural directions follow protocol | Metric direction |

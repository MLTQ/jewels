# `aggregate_encoder_convergence.py`

## Purpose

Turns the corrected multi-size, multi-seed encoder matrix into the primary quantitative scaling
claim and a pitch-readable convergence figure.

## Components

### `confidence`
- Reports sample mean/standard deviation and a two-sided 95% Student-t interval for three seeds.

### `main`
- Reads best frozen-validation checkpoints, retains per-seed and per-style results, aggregates
  matched-epoch learning curves, and reports the largest-two-budget delta. Early-stopped seeds are
  carried forward at their final selected checkpoint, so every displayed curve point retains the
  complete seed count instead of silently narrowing late in training.
- Writes `report.json` plus a two-panel PNG of data scaling and convergence over corpus passes.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Feasibility report | Seed-level results remain visible beneath aggregate intervals | Report schema |
| Pitch visual | Axes identify videos, epochs, PSNR, and 95% interval meaning | Plot semantics |
| Curve reader | Every reported epoch contains all declared seeds | Early-stop alignment |

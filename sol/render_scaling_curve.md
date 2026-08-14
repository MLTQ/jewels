# `render_scaling_curve.py`

## Purpose

Turns the scaling experiment's committed artifacts into the curve the compute pitch rests on:
held-out velocity loss, autonomous-rollout PSNR, and rollout LPIPS as functions of the number of
class-balanced fitted training sources, at a frozen recipe/protocol/seed.

## Components

### `collect_point`
- **Does**: Extracts one curve point from a mark-flow training `summary.json`
  (`latest_evaluation.aggregate`) and a `perceptual-arm-eval-v1` report macro for a named arm,
  keeping the correct/shuffled margin alongside for a conditioning sanity check.

### `render_curve`
- **Does**: Three small-multiple panels over log2 source count, one hue, direct point labels,
  no dual axes; each panel titles which direction is better.
- **Rationale**: The three metrics live on incompatible scales, so they get separate panels
  rather than shared or twin axes.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Paper scaling section | Figure PNG plus `realizer-data-scaling-v1` table JSON | Schemas |
| `sol/results/*/data_scaling*` | Points recomputable from committed summaries/reports | Input formats |

# `trajectory_tube_realizer.py`

## Purpose

Builds a novel Jewel field from two distinct training programs: one persistent token owns a moving
foreground tube and another owns the surrounding scene. This is the compositional successor to the
Gate 2a8 whole-source coherence ceiling.

## Components

### `TrajectoryTubeRealizer`

- **Does**: Ranks coherent donors by addressed block-program distance, requires distinct foreground
  and background sources, and computes a fixed-radius temporally smoothed tube from donor
  disagreement.
- **Does**: Casts foreground Jewels inside the tube and background Jewels outside, then performs a
  matched uniform count adjustment.
- **Rationale**: All blocks in the subject tube share one source owner, retaining cross-block and
  cross-time dependencies that independent local phrase selection destroyed.

### `fit_trajectory_tube_realizer`

- **Does**: Reuses the train-owned coherent source representation and exposes its exact active-token
  fields to the tube compositor.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a9 audit | Two distinct donors and fixed radius 0.78 | Composition semantics |
| Novelty claim | Both donors contribute materially; no target Jewel rows are copied | Source ownership |
| Future speaker | Scene token, tube/track token, background token, then local Jewel phrases | Grammar shape |

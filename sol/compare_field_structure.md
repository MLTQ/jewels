# `compare_field_structure.py`

## Purpose

Measures whether fitted and encoded jewel fields form content-clustered spacetime tubes rather
than uniform raster-like blobs.

## Components

### `structure_report`
- **Does**: Applies the canonical 2% opacity floor, then reports anisotropy, extent dispersion,
  occupancy entropy, temporal alignment, and mixed spacetime tilt.
- **Does**: Defines mixed tilt as `2 |v_t| sqrt(1 - |v_t|^2)`, which is zero for purely spatial or
  purely temporal axes and one for a 45-degree spacetime trajectory.
- **Does**: Solves covariance eigensystems in bounded float32 chunks. This preserves the precision
  of canonical float32 features while avoiding multi-gigabyte CUDA solver workspaces for a batch
  of many tiny 3x3 matrices.

### `main`
- **Does**: Loads paired fitted fields and videos, evaluates the encoder on matching sources, and
  writes per-source plus macro structure records.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Structural trainers/audits | Canonical 22-D features and 2% active floor | Feature/threshold semantics |
| Feasibility gate | Mixed tilt distinguishes actual trajectories from arbitrary elongation | Metric definition |
| GPU audits | `sample` and `eigen_chunk` are positive, bounded work budgets | Default budget semantics |

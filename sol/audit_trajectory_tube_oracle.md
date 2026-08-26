# `audit_trajectory_tube_oracle.py`

## Purpose

Runs Gate 2a9: a source-disjoint, two-donor compositional oracle in which one persistent token owns
a connected moving foreground tube and another owns the surrounding scene.

## Components

### `evaluate_source_disjoint`

- **Does**: Renders target, correct two-donor composite, coherent-source ceiling, wrong-scene
  foreground, and pooled-null controls at matched points and random seeds.
- **Does**: Measures active-token histogram agreement, voxel PSNR diagnostics, donor contribution,
  count adjustment, structural retention, and grid locking.

### `main`

- **Does**: Loads the immutable split/codebooks, fits only train-owned tube state, and writes the
  frozen report plus qualitative contact sheet.
- **Rationale**: A recognizable composite rules out complete-video retrieval as the only way to
  preserve coherence, while the wrong-object arm tests semantic ownership of the tube.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a9 protocol | Nearest and next-nearest same-scene donors are distinct | Donor selection |
| Novelty audit | Target supplies only its block program; emitted rows are training-owned | Leakage boundary |
| Qualitative review | All arms share renderer, background, frames, and seeds | Comparison parity |

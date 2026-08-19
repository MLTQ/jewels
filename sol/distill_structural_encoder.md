# `distill_structural_encoder.py`

## Purpose

Render loss alone drives the structural encoder toward a uniform lattice — measured across
training: extent IQR fell 2.10 -> 1.35 and occupancy uniformity rose 0.9961 -> 0.9992 while
PSNR climbed monotonically. The fitter escapes that optimum only because densify/prune is an
explicit structural mechanism outside the loss. This distils the fitter's structure into the
feed-forward student, which is what PROMPTABLE_ROADMAP Phase 3 specified from the start.

## Components

### `teacher_descriptors`
- **Does**: Opacity-weighted subsample of a fitted field, returning centres and each jewel's
  log-eigenvalue **spread** (a scale-invariant anisotropy descriptor).
- **Rationale**: The student holds 10k jewels against the teacher's 72k, so its primitives must
  be larger; matching absolute size would be wrong, matching shape character is not.

### `chamfer`
- **Does**: Symmetric squared-distance Chamfer plus student->teacher nearest indices.
- **Rationale**: The teacher->student direction is the one that forces clustering — every region
  the fitter densified must have a student near it. Student->teacher keeps students off empty
  space. Tested explicitly for the uncovered-cluster case.

### `main`
- **Does**: Trains with render loss plus weighted Chamfer and spread-matching terms, reporting
  the same structure battery as the render-only run for a controlled comparison.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Path B comparison | Same structure metrics as `train_structural_encoder.py` | Report schema |
| Teacher fields | Fitted checkpoints named `<video stem>_w000000.pt` | Naming |

## Notes

- Runs on `ltx_domain_v1` (12 train + 4 validation) because only those windows have fitted
  fields; the 240-clip diverse corpus is unfitted. Small, but the question is mechanistic.

# `structural_encoder.py`

## Purpose

Path B (`jewels-gn8`): an encoder whose jewels must **describe** content rather than sample it.
The lattice encoder's learned features contributed only 0.15 dB at inference because its
quality came from copying video colours onto a fixed grid — a re-encoded video, which cannot
support jewels-as-tokens. This encoder removes the crutch and gives shape enough freedom to
form the sheared spacetime tubes that are the project's founding premise.

## Departures from `amortized_encoder.VideoToJewelEncoder`

1. **Scarcity** — 10,240 jewels/window (2,048 cells x 5 slots) versus 73,728, sized from the
   image-splatting literature's 5-10k per still adjusted for temporal coherence. A primitive
   must stretch to cover structure instead of tiling a patch.
2. **No content lookup** — colours are pure network outputs; nothing samples the video.
3. **Tube-capable shape** — quaternion plus three independent log scales (the fitter's own
   parameterization) instead of a near-diagonal Cholesky whose `0.2 *` off-diagonals capped
   anisotropy near 2.
4. **Free positions** — centres may migrate +/-2 cell extents from their anchor, so density can
   follow content rather than being pinned to a grid.

Geometric initialization is *retained* (lattice anchors, coverage-calibrated scales): the
v0 lesson was that geometry init matters, and the fork is about content lookup, not init.

## Components

### `quaternion_to_matrix` / `precision_factor`
- **Does**: Builds `M` with `precision = M M^T` as `R diag(exp(-s))`. The renderer evaluates
  `||M^T d||^2`, which needs only `M M^T` to equal the precision — `M` need not be triangular —
  so arbitrary anisotropy stays expressible.

### `StructuralJewelEncoder`
- **Does**: 3D-conv trunk to the cell grid, then per-slot centre offset, quaternion, log scales,
  colour, colour gradient, and opacity. `canonical_features` emits the standard 22-D layout so
  every existing audit tool applies unchanged.

### `render_structural`
- **Does**: Additive render matching `sol.render.render_exact` semantics, unit-verified, with
  gradient checkpointing per point block.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `train_structural_encoder.py` | Prediction dict keys incl. `precision_factor` | Output contract |
| `compare_field_structure.py` | Canonical 22-D features | Feature layout |

## Notes

- Targets from `results/field_structure_v1`: anisotropy median ~10, extent IQR ~2.3, occupancy
  uniformity <= ~0.95. Known failure mode: uniform blobs are the safe optimum under plain L2, so
  collapse back toward the lattice is the result to watch for.

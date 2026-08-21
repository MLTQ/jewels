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
2. **Optional continuous colour seeding** — a declared arm can seed colour from the video at
   each predicted mobile centre. Unlike the baseline, this does not pin colour to a fixed slot
   lattice; the later generative model must emit the equivalent colour information.
3. **Tube-capable shape** — quaternion plus three independent log scales (the fitter's own
   parameterization) instead of a near-diagonal Cholesky whose `0.2 *` off-diagonals capped
   anisotropy near 2.
4. **Free positions** — centres may migrate +/-2 cell extents from their anchor, so density can
   follow content rather than being pinned to a grid.
5. **Versioned irregular proposals** — new checkpoints use deterministic irrational-rotation
   offsets that remain inside every cell for arbitrary slot counts. This avoids the legacy
   36-slot cube-rounding duplication without changing old lattice checkpoints.
6. **Optional video-seeded colour** — the irregular-field gate may seed colour at the predicted
   continuous centres to preserve the successful amortized encoder's fidelity. This is declared in
   checkpoint metadata because a later text prior must generate the seed information.

Geometric initialization is *retained* (lattice anchors, coverage-calibrated scales): the
v0 lesson was that geometry init matters, and the fork is about content lookup, not init.

## Components

### `quaternion_to_matrix` / `precision_factor`
- **Does**: Builds `M` with `precision = M M^T` as `R diag(exp(-s))`. The renderer evaluates
  `||M^T d||^2`, which needs only `M M^T` to equal the precision — `M` need not be triangular —
  so arbitrary anisotropy stays expressible.

### `StructuralJewelEncoder`
- **Does**: 3D-conv trunk to the cell grid, then per-slot centre offset, quaternion, log scales,
  colour, colour gradient, and opacity. Centre mobility is configurable in cell extents, and colour
  may optionally be sampled at the predicted continuous centres. `canonical_features` emits the
  standard 22-D layout so every existing audit tool applies unchanged.

### `stratified_slot_offsets`
- **Does**: Produces any requested number of deterministic, non-grid proposal anchors strictly
  inside a cell.
- **Rationale**: A rounded cubic side silently wraps and duplicates positions whenever the slot
  count exceeds the rounded cube's capacity (including the old 36-slot configuration).

### `render_structural`
- **Does**: Delegates arbitrary rotation/scale precision factors to the exact or support-complete
  tiled training renderer, retaining the canonical additive/P1 semantics and gradient checks.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `train_structural_encoder.py` | Prediction dict keys incl. `precision_factor` | Output contract |
| `compare_field_structure.py` | Canonical 22-D features | Feature layout |
| Irregular-field trainer | Seed-colour and mobility choices survive checkpoint metadata | Model arguments |

## Notes

- Targets from `results/field_structure_v1`: anisotropy median ~10, extent IQR ~2.3, occupancy
  uniformity <= ~0.95. Known failure mode: uniform blobs are the safe optimum under plain L2, so
  collapse back toward the lattice is the result to watch for.

# `distill_structural_encoder.py`

## Purpose

Render loss alone drives the structural encoder toward a uniform lattice — measured across
training: extent IQR fell 2.10 -> 1.35 and occupancy uniformity rose 0.9961 -> 0.9992 while
PSNR climbed monotonically. The fitter escapes that optimum only because densify/prune is an
explicit structural mechanism outside the loss. This distils the fitter's structure into the
feed-forward student, which is what PROMPTABLE_ROADMAP Phase 3 specified from the start.

## Components

### `teacher_descriptors`
- **Does**: Opacity-weighted subsample of a fitted field, returning centres, each jewel's
  log-eigenvalue **spread** (a scale-invariant anisotropy descriptor), principal axes, and
  opacity weights, plus mean log scale as an absolute extent target.
- **Rationale**: The student holds 10k jewels against the teacher's 72k, so its primitives must
  be larger; matching absolute size would be wrong, matching shape character is not.
- **Update**: The 40,960-proposal v3 arm measured median extent `0.0516` against teacher `0.01285`
  despite matched anisotropy. `--size-weight` and an explicitly declared log `--size-offset` now
  test whether excessive absolute extent—not proposal count—is the causal blur source.

### `principal_axis` / `orientation_loss`
- **Does**: Extracts the rotation column belonging to the largest scale (the direction a tube
  points) and scores `1 - |cos|` against the nearest teacher's principal axis; absolute because
  an axis has no sign.
- **Rationale**: Supervising spread alone produced elongation *without direction* — visible in
  contact sheets as horizontal smearing rather than object-tracking tubes. Anisotropy measured
  6.29 while the renders streaked, so magnitude was learned and purpose was not.
- **Scheduling**: Axis pressure has its own delayed linear ramp. The irregular seed-0 audit measured
  anisotropy `5.48` but mixed spacetime tilt only `0.073`, proving that elongation without a strong
  direction term becomes horizontal smear rather than a time-distorted tube.

### `mixed_spacetime_tilt`
- **Does**: Computes the gate quantity directly from each principal axis and allows a smooth-L1
  student/teacher tilt match with `--tilt-weight`.
- **Rationale**: Raising nearest-axis cosine supervision 5× left held-out mixed tilt unchanged
  (`0.084`); the indirect loss can be low while the field still chooses nearly pure spatial or
  temporal axes. Direct supervision tests the actual trajectory property instead.

### `soft_occupancy` / `density_loss`
- **Does**: Differentiably bins jewel centres into per-cell occupancy shares (softmax over
  distance to cell centres, so gradients flow to positions), weights both fields by opacity, and
  scores symmetric KL between the student's and teacher's distributions. Enabled with
  `--density-weight`.
- **Rationale**: Chamfer taught shape but not placement — measured across v1 and v2, occupancy
  uniformity stalled near 0.996 against the fitter's 0.946 while every shape metric improved.
  With enough students the nearest-teacher distance is already small everywhere, so Chamfer has
  no gradient pushing mass into dense regions; matching binned densities penalises exactly the
  "too few jewels where the fitter put many" error that correspondence losses miss.
- **Scheduling**: Density pressure can use its own delayed linear ramp. The first selected 6k arm
  proved fidelity/sparsity but stopped at `0.9894` occupancy uniformity because its `0.2 * KL`
  contribution fell below `0.002`; the exact training and validation teachers measured `0.9610`
  and `0.9646`, so a stronger delayed arm tests a real remaining gap rather than a stale target.

### `soft_active_fraction`
- **Does**: Smoothly counts proposals above the canonical 2% opacity floor so a declared target
  active fraction and gate-polarization penalty can train actual sparsity.
- **Rationale**: A fixed proposal grid is harmless only if unused proposals become inactive; the
  lattice encoder's measured 100% active fraction showed ordinary render loss does not do it.

### `schedule_multiplier`
- **Does**: Delays and linearly ramps sparsity/polarization pressure while render supervision first
  establishes colour and shape.
- **Rationale**: The matched 200-step screens measured the causal tradeoff: immediate sparsity
  de-gridded the field (`0.9788` uniformity, `0.540` active) but cost 1.90 dB versus the 100%-active
  control. Scheduling preserves the necessary mechanism without starving reconstruction at warm-up.

### `freeze_geometry_state` / `mask_geometry_gradients` / `restore_geometry_state`
- **Does**: Freezes the shared trunk, masks the mixed head's center/quaternion/scale/opacity rows,
  and restores those rows after AdamW so decoupled weight decay cannot move them.
- **Rationale**: The direct-tilt arm passed every geometry threshold at step 2,000, then moved back
  toward uniform coverage while gaining fidelity. An exact appearance-only continuation tests
  whether colour, colour-gradient, and background learning can recover detail without sacrificing
  that already-successful geometry state.

### `multiscale_image_loss`
- **Does**: Scores full low-resolution frame renders with a three-level Charbonnier pyramid and
  horizontal/vertical first-order edge differences.
- **Rationale**: Random voxel MSE improved sampled PSNR while leaving broad, perceptually poor
  renders. A bounded image-grid term gives the appearance branch spatial context and explicit edge
  pressure without changing the exact held-out LPIPS gate.

### Local teacher attributes
- **Does**: Uses detached soft correspondence from `local_teacher_distillation.py` to supervise
  ordered local scales, sign-invariant axes, opacity optical mass, base color, and color gradients.
- **Rationale**: Factorized v3 learned irregular placement but rendered broad misplaced color.
  Global density/spread/size statistics cannot tell the student which covariance and appearance
  belong to a particular content region.
- **Scheduling**: Local losses have an independent delayed ramp and separately logged weights.

### Renderer-weighted teacher responsibilities
- **Does**: Uniformly samples active fitted jewels with an independent RNG, then asks
  `local_teacher_distillation.py` for opacity/Mahalanobis contribution moments at the same bounded
  student subset used by the structural losses.
- **Does**: Applies separately scheduled scale, axis, optical-density, local rendered-color, and
  local color-Jacobian losses, while logging support and effective responsibility counts.
- **Rationale**: Position-only raw attributes improved LPIPS on all five styles but lost PSNR even
  after equal relaxation. Responsibility moments preserve the signal while making the target about
  the jointly rendered local field rather than an arbitrary nearest fitted jewel.
- **Matching**: The active-uniform sample has its own CPU generator so enabling this arm cannot
  perturb the opacity-sampled teacher set or GPU training sequence used by its matched control.
- **Appearance contract**: `--appearance-contract residual` expands a bounded checkpoint with a
  zero-initialized unconstrained head. `--responsibility-appearance-target raw` is accepted only for
  that contract, preventing an impossible raw target from being paired with the bounded encoder.

### `chamfer`
- **Does**: Symmetric squared-distance Chamfer plus student->teacher nearest indices.
- **Rationale**: The teacher->student direction is the one that forces clustering — every region
  the fitter densified must have a student near it. Student->teacher keeps students off empty
  space. Tested explicitly for the uncovered-cluster case.

### `main`
- **Does**: Loads a manifest with partial fitted-teacher coverage, oversamples declared teacher
  examples without hiding broad render training, and combines support-complete render, Chamfer,
  spread/orientation, opacity-weighted density, sparsity, and polarization terms.
- **Does**: Retains the corpus on CPU and stages one source at a time, so corpus size does not consume
  accelerator memory and smaller local GPUs can run the same declared protocol.
- **Does**: Allows an explicit ordered validation subset for bounded screens while leaving the full
  training corpus unchanged; unknown source IDs are rejected instead of silently substituted.
- **Does**: Records colour seeding, centre mobility, renderer capacity, teacher sampling, and
  sparsity settings in the checkpoint.
- **Does**: Preserves every numbered evaluation checkpoint as well as `encoder.pt`, so a
  multi-objective audit cannot silently lose an intermediate result.
- **Does**: Can initialize a declared continuation only from an architecture/grid/model-compatible
  checkpoint; the source checkpoint is recorded in descendant metadata.
- **Does**: With `--freeze-geometry`, trains only colour, colour-gradient, and background parameters;
  correspondence computation is skipped when every teacher-structure weight is zero.
- **Does**: Selects the versioned factorized-v3 architecture without changing v2 checkpoint
  semantics, records its appearance dimensions, and can transplant only compatible v2 geometry.
- **Does**: Factorized freezing uses ordinary module ownership; v2 retains its exact mixed-row
  masking/restoration path for backward-compatible causal controls.
- **Does**: Optionally renders a deterministic rotating subset of low-resolution full frames every
  declared number of steps and records the image-grid loss separately from sampled-voxel MSE.
- **Does**: Optionally matches nearest-teacher absolute mean log scale in addition to scale-invariant
  spread; the separate offset declares any coverage compensation for a lower active jewel count.
- **Does**: Optionally applies local fitted-teacher scale/axis/optical-mass/RGB/RGB-gradient losses
  over the same bounded student subset. Correspondence is detached from centers, and opacity mass
  is compensated by full-teacher active count divided by the declared student target count.
- **Does**: Optionally applies renderer-responsibility moment losses from a separate active-uniform
  teacher sample, with independent support, temperature, size offset, schedule, weights, and logs.
- **Does**: Explicitly transfers a legacy bounded v3 checkpoint into the shape-compatible residual
  contract only when every pre-existing model argument matches.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Path B comparison | Same structure metrics as `train_structural_encoder.py` | Report schema |
| Teacher fields | Source-owned support-correct checkpoints or legacy `<video stem>_w000000.pt` fits | Naming |
| Prompt-prior gate | `structural_jewel_encoder_v2` identifies sparse irregular checkpoints | Architecture ID |
| Frozen-geometry continuation | Center, covariance, and opacity predictions remain bitwise fixed for each video | Freeze mask |
| Factorized-v3 checkpoints | Architecture-specific constructor arguments and optional v2 geometry source are recorded | Metadata |
| Appearance-grid objective | Positive dimensions/frequency and an explicitly weighted loss | Training semantics |
| Absolute-size experiment | Teacher mean log scale and declared offset retain physical scale meaning | Descriptor schema |
| Local attribute experiment | Weights, neighbor kernel, schedule, and active-count compensation are checkpointed | Local-loss semantics |
| Responsibility experiment | Active sample, finite support, temperature, offset, weights, schedule, and diagnostics are checkpointed | Responsibility semantics |
| Residual appearance experiment | Source/target contracts and raw-vs-bounded responsibility targets are checkpointed | Appearance semantics |

## Notes

- A failure on limited teacher coverage narrows the declared weighting/capacity configuration; it
  does not rule out irregular amortized fields. Positive arms must be replicated before selection.

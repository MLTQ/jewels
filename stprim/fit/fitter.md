# fitter.py

## Purpose
Per-clip stochastic-voxel fitting loop. This is stage 1 — the same stage GSVC/VeGaS occupy. It
exists to produce training data for the amortized/generative stage, so it's tuned for throughput
and reproducibility over last-dB quality.

## Components

### `FitConfig`
- **Does**: all hyperparameters for a fit, including explicit isotropic versus temporal-preserving
  spatial densification and the renderer's culling contract
- `cull_mode="knn"` preserves historical behavior. `"support"` uses a declared finite Gaussian
  support and fails on candidate overflow; `"exact"` exists only for tiny correctness probes.
- `support_sigma`, `support_capacity`, and `support_point_chunk` are checkpointed, so recovery and
  downstream evaluation cannot silently change the renderer used by a fit.
- `geometry_constraint` is `free` for the actual representation. `axis_aligned` projects every
  quaternion to identity while retaining distinct spatial/temporal scales; `isotropic` also ties
  the three scales. Both are experiment controls for measuring the causal value of spacetime tilt.

### `project_geometry_(field, constraint)`
- **Does**: applies the selected causal geometry projection after initialization/resume, every
  optimizer step, and every densification event.
- **Rationale**: projecting only at evaluation would test checkpoint damage, not whether the fitter
  can compensate under the constrained representation. Projected training gives each control its
  best fit at the same optimizer and primitive budget.

### `fit_volume(video, cfg, device, resume_state, checkpoint_every, checkpoint_callback)`
- **Does**: (T,H,W,3) in [0,1] -> (PrimitiveField, info dict)
- **Interacts with**: `make_grid`, `sample_indices`, `render_points`, `adapt`
- **Recovery**: after a completed optimizer step (including logging and adaptation), optionally
  emits a CPU snapshot containing the next step, field/background, Adam state, gradient tracker,
  history, elapsed time, and the dedicated fit RNG state

## Decisions
- Per-parameter-group learning rates: geometry (mu/scale/quat) at half the appearance LR.
  Geometry moving as fast as color destabilizes early fitting.
- MSE only. Rejected adding SSIM/gradient loss for now — extra hyperparameters obscure the
  representation question.
- PSNR reported is computed on the sampled voxel batch, so it's noisy and NOT comparable to
  published full-frame PSNR. It's a training signal, not a benchmark number.
- Optimizer rebuilt after every adapt (see `adapt.md`).
- **[2026-08-04] Exact recovery**: stochastic voxel sampling and densification noise use one
  dedicated CPU generator. This costs one small index/noise transfer per step, but makes a saved
  RNG state portable across CUDA, MPS, and CPU. The callback runs only at a fully completed step
  boundary, so it never exposes a half-updated field.
- **[2026-08-04] Spatial split control**: the opt-in split policy preserves the most time-aligned
  principal scale. The default remains isotropic so historical checkpoints and recovery probes do
  not silently change meaning.
- **[2026-08-19] Culling is an experimental variable**: fitting now forwards the complete renderer
  policy from `FitConfig`. Support-mode candidate overflow is fatal by design; increasing capacity
  and restarting is preferable to training on an unknown, geometry-dependent truncation.
- **[2026-08-19] Geometry constraints are causal controls**: free, axis-aligned, and isotropic arms
  share the fitter. The constraints are stored in checkpoints/recovery configs and are enforced
  after adaptation so split children cannot escape the ablation.
- **[2026-07-31]** The voronoi mode and its steelman knobs (tau anneal, bg pseudo-cell, Lloyd)
  were removed with the branch — see PROJECT.md decision log. `FitConfig` no longer has a
  `mode` field; checkpoints made before this date carry one.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `cli/fit_video.py`, `cli/render_recon.py` | `(field, info)` with `info["history"]`, `info["background"]`, `info["shape"]` | info schema — freeze before corpus generation |
| `experiments/canonicalization.py` | `FitConfig(seed=...)` fully determines the run | Seeding behaviour |
| `cli/fit_corpus.py` | recovery state version 1 and callback at `next_step` boundaries | Recovery state schema |

## Notes
- The learned background is optimized jointly but lives outside `PrimitiveField`, so
  `fit_volume` returns it in `info["background"]` (a 3-float list — info is JSON-serialized by
  the CLIs). Anything reconstructing from a saved fit MUST add it back; `state_dict()` alone is
  not the full model.
- Recovery snapshots are callback-driven: `fit_volume` owns numerical state while the caller
  chooses persistence policy. `cli/fit_corpus.py` uses atomic files; tests can keep snapshots in
  memory.

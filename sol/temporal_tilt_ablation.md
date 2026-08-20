# `temporal_tilt_ablation.py`

## Purpose

Tests the project's defining causal claim: at fixed fitting compute and primitive budget, freely
tilted spacetime Gaussians should reconstruct moving video better than Gaussians whose axes are
projected onto space/time coordinates.

Interpret conclusions under `sol/EVIDENCE_POLICY.md`: one matched source can establish a causal
signal, but replication is required before a decision-grade claim.

## Experiment

- All arms use the support-correct five-sigma renderer.
- `free` learns scales and rotations normally.
- `axis_aligned` retains independent scales but projects every quaternion to identity after
  initialization, every optimizer step, and densification. It can model temporal extent but cannot
  express a sheared motion tube with one primitive.
- `isotropic` is an optional stronger control that also ties all three scales.
- Seed, sampled voxels, optimizer steps, initialization count, maximum primitive count, and
  adaptation schedule are matched.
- `--seeds` runs multiple paired initializations in seed-first order, so every completed free arm
  is followed immediately by its projected control. The incremental v2 report and seed-qualified
  checkpoints make a killed batch resumable without discarding completed pairs.
- Every real source is bound to its resolved path, byte size, and SHA-256 digest. Resume rejects a
  file that changed in place, and aggregation retains the fingerprint as provenance.

## Metrics and gate

Every arm reports support-evaluated full-volume PSNR, field structure, raw tensor storage, source
voxels per primitive, and descriptive PSNR-per-budget ratios. Error reporting includes global RGB
MAE, p95/max pixel error, worst-frame error, and RGB MAE in the target video's top-20% motion
regions. Mixed spacetime tilt is `2 t sqrt(1-t²)` for the longest principal axis, where `t` is its
absolute temporal component; it is zero for purely spatial or temporal axes and one for a balanced
diagonal.

The per-source summary uses paired Student-t uncertainty across seeds. The predeclared causal gate
requires the largest-budget free mean to beat the axis-aligned control by at least 0.5 dB, use
median mixed tilt of at least 0.2, and verify every control's median mixed tilt is numerically zero.

### `mean_confidence_interval(values)`
- **Does**: paired mean, sample standard deviation, and two-sided 95% Student-t interval.
- **Rationale**: small seed counts must not inherit a normal approximation or hide dispersion.

### `reconstruction_error(prediction, target)` / `field_storage(field, target)`
- **Does**: local-error and quality-at-budget evidence shared by every causal arm.

### `source_fingerprint(path)`
- **Does**: records resolved source path, byte size, and SHA-256 before fitting.

### `environment_report(device)`
- **Does**: records PyTorch version, resolved device, accelerator model, and accelerator UUID in
  every report so CUDA/nvidia-smi ordering cannot misattribute hardware again.

### `write_report(path, report)`
- **Does**: atomically replaces the incremental JSON report for interruption safety.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment reports | `temporal-tilt-ablation-v2` paired seed records, protocol, local error, storage, and causal gate | Schema or metric definitions |
| Stage-1 fitter | Enforces `geometry_constraint` throughout training | Projection timing or semantics |

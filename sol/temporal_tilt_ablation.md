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

## Metrics and gate

Every arm reports support-evaluated full-volume PSNR and field structure. Mixed spacetime tilt is
`2 t sqrt(1-t²)` for the longest principal axis, where `t` is its absolute temporal component; it
is zero for purely spatial or temporal axes and one for a balanced diagonal.

The predeclared causal gate requires the largest free arm to beat the axis-aligned control by at
least 0.5 dB, use median mixed tilt of at least 0.2, and verify the control's median mixed tilt is
numerically zero.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment reports | `temporal-tilt-ablation-v1` records and causal gate | Schema or metric definitions |
| Stage-1 fitter | Enforces `geometry_constraint` throughout training | Projection timing or semantics |

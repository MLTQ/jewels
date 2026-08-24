# Residual appearance contract v1 — frozen protocol

## Question

Does removing the factorized-v3 per-jewel sigmoid RGB and `0.25*tanh` Jacobian bounds allow
renderer-responsibility appearance targets to improve both exact PSNR and LPIPS while the learned
field remains irregular, sparse, and tilted through spacetime?

## Source and causal transfer

- Architecture remains `factorized_structural_jewel_encoder_v3` with 40,960 proposals
  (`16 x 16 x 8 x 20`). The constructor records `appearance_contract=residual`.
- Source is `local_teacher_distillation_v1/relaxation/control_total1200/encoder.pt`, the same bounded
  checkpoint used by responsibility v1: sampled PSNR `18.5603 dB`, exact `18.3653 dB / 0.76791`
  LPIPS, occupancy `0.98202`, active fraction `0.64384`, and mixed tilt `0.52159`.
- Every source parameter loads unchanged. A new 12-output residual appearance head is initialized to
  exactly zero after the bounded RGB/Jacobian prediction. Tests require every initial prediction,
  canonical feature, and render input to be bitwise identical to the bounded source.
- The residual head can emit unconstrained RGB and RGB Jacobians. Geometry, covariance, opacity, and
  background parameterizations are unchanged.

## Read-only calibration

`calibration_raw.json` uses the declared source, seed 0, 4,000 active-uniform fitted teachers,
1,024 opacity-sampled students, five-sigma support, and 1,024 render points without constructing an
optimizer.

| Component | Raw loss | Geometry gradient | Appearance gradient |
|---|---:|---:|---:|
| sampled render | 0.01914 | 0.17221 | 0.06938 |
| raw response color | 0.14491 | 0 | 1.33007 |
| raw response Jacobian | 1.90608 | 0 | 0.10480 |

The teacher support statistics exactly reproduce responsibility v1: mean support `6.52`, effective
contributors `1.73`, and fallback `0.586%`. Although `44.1%` of color and `75.5%` of Jacobian
elements lie outside the old head bounds, the residual/raw contract projects `0%` of either target.
At weights `0.025 / 0.025`, the separately measured response gradients are approximately 52% of the
expanded render appearance-gradient norm, close to the prior bounded arm's approximately 46%.

## Registered arms

Both arms start from the exact same zero-expanded source and receive 600 continuation steps, seed 0,
the same 120 videos, 36 fitted training teachers, optimizer, learning-rate schedule, render samples,
and existing structural losses as responsibility v1.

| Arm | Appearance contract | Response RGB | Response Jacobian | Target projection |
|---|---|---:|---:|---|
| residual-control | residual | 0 | 0 | none used |
| residual-response | residual | 0.025 | 0.025 | raw / none |

Response losses start after step 100 and linearly reach full weight at step 400. All response
geometry and opacity weights and all position-only local losses remain zero. This is not another
interpolation of the rejected bounded/local weights.

Shared structural losses remain Chamfer `0.2`, spread `0.02`, absolute size `0.023` with log offset
`+0.35`, orientation `0.1`, direct tilt `1.0`, density `2.0`, sparsity `0.1`, polarization `0.01`,
and target active fraction `0.58`. Training uses AdamW peak LR `1e-4`, warmup 100, 1,024 sampled
render points, 1,024 sampled students, and support-tiled capacity 8,192.

## Decision rules

An arm is exact-audited only if sampled occupancy is `<=0.985`, active fraction is `<=0.70`, mixed
tilt is `>=0.25`, PSNR is no more than `0.50 dB` below residual-control, and training remains finite.

The raw-response objective passes causally only if residual-response improves both exact macro PSNR
and LPIPS over residual-control with all structure gates retained. The residual contract passes as a
representation if at least one residual arm improves both metrics over the unchanged bounded control
(`18.50932 / 0.75795`) and the frozen bounded midpoint (`18.50301 / 0.75308`). Contact sheets must
also show a defensible contrast or boundary improvement; a metric-only sub-visual change is recorded
but not promoted.

Absolute promotion remains exact PSNR `>=20 dB` and LPIPS `<=0.40`. A joint representation pass
below that gate warrants a longer matched continuation and unchanged seeds 1–2; otherwise no seed
replication is run. Failure rejects only this zero-residual parameterization, raw composited target,
weights, source checkpoint, seed, and 600-step schedule—not unbounded splat appearance as a class.

## Compute ownership

The runs use the released RTX 2070 Super. `scroller-llm.service` was cleanly stopped before the
experiment and must be restored at session end. Wall time on the shared machine is operational data,
not evidence about model scaling.

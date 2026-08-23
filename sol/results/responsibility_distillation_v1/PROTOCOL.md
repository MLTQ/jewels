# Renderer-responsibility distillation v1 — frozen protocol

## Question

Can opacity- and covariance-support-weighted fitted-teacher moments preserve the five-style LPIPS
signal from local distillation while also improving exact PSNR, without restoring the proposal
lattice or violating sparsity/time-distortion gates?

## Source and shared controls

- Architecture: `factorized_structural_jewel_encoder_v3`, 40,960 proposals (`16 x 16 x 8 x 20`).
- Source checkpoint: `local_teacher_distillation_v1/relaxation/control_total1200/encoder.pt`.
- Source exact audit: `18.3653 dB`, `0.76791` LPIPS; sampled held-out PSNR `18.5603 dB`,
  occupancy `0.98202`, active fraction `0.64384`, mixed tilt `0.52159`.
- Data: the same 120 training videos, 36 source-owned fitted training fields, and five held-out
  evaluation styles used by the previous v3 gates.
- Compute: seed 0, 600 continuation steps, AdamW peak LR `1e-4`, warmup 100, 1,024 sampled render
  points/step, 1,024 opacity-sampled student moment points, 4,000 active-uniform fitted teachers,
  support-tiled renderer capacity 8,192.
- Existing structure losses remain unchanged: Chamfer `0.2`, spread `0.02`, absolute size `0.023`
  with log offset `+0.35`, orientation `0.1`, direct tilt `1.0`, density `2.0`, sparsity `0.1`,
  polarization `0.01`, target active fraction `0.58`.
- Responsibility kernel: canonical opacity times Gaussian density under covariance precision,
  five-sigma finite support, temperature `1.0`, active-uniform teacher sampling, and detached student
  centers. Queries without a sampled supported teacher use minimum-Mahalanobis fallback and are
  counted. Moment losses start after step 100 and reach full weight at step 400.
- Responsibility scale offset is `0.0`: the mixture covariance already includes within-teacher and
  between-teacher spread. Feasible-target projection is fixed to the v3 output contract.

## Read-only calibration

`calibration.json` was produced without an optimizer on the declared source checkpoint and
`anime__00_cooking_train_00_seed60000`. Raw loss and parameter-gradient norms were:

| Component | Loss | Geometry gradient | Appearance gradient |
|---|---:|---:|---:|
| sampled render | 0.01914 | 0.17221 | 0.01294 |
| responsibility scale | 0.63015 | 3.68123 | 0 |
| responsibility axis | 0.36218 | 0.27232 | 0 |
| responsibility opacity | 0.45941 | 0.80975 | 0 |
| responsibility color | 0.10104 | 0 | 0.20771 |
| responsibility Jacobian | 0.03010 | 0 | 0.02924 |

The teacher sample provides 6.52 supported candidates and 1.73 effective contributors per student
on average; only 0.586% require fallback. Before feasible projection, 44.1% of color elements and
75.5% of Jacobian elements lie outside the student's bounds. The registered weights are therefore
substantially lower than the rejected raw-attribute weights.

## Registered arms

| Arm | Resp. scale | Resp. axis | Resp. opacity | Resp. RGB | Resp. Jacobian |
|---|---:|---:|---:|---:|---:|
| control | 0 | 0 | 0 | 0 | 0 |
| response-appearance | 0 | 0 | 0 | 0.025 | 0.025 |
| response-full | 0.005 | 0.025 | 0.01 | 0.025 | 0.025 |

At calibration, the full arm adds approximately 19% of the render geometry-gradient norm and 46%
of its appearance-gradient norm before scheduling. No registered arm uses the previous position-only
local losses.

## Screen and exact decision

An intervention remains eligible only if sampled held-out occupancy is `<=0.985`, active fraction
is `<=0.70`, mixed tilt is `>=0.25`, and PSNR is no more than `0.50 dB` below the matched control.
Every eligible intervention is exact-audited beside the control under one process.

The responsibility hypothesis passes only if one arm improves **both** exact PSNR and LPIPS over
control while retaining all three structural gates. Promotion still requires absolute exact PSNR
`>=20 dB` and LPIPS `<=0.40`; only a promoted arm is replicated at seeds 1 and 2. A failed arm
rejects only this sample, support, feasible projection, weight, source checkpoint, seed, and 600-step
schedule—not renderer-weighted targets as a class.

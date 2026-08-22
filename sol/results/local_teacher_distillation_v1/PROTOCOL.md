# Local fitted-teacher distillation v1 — frozen protocol

## Question

Can assigning fitted-teacher covariance and appearance to nearby irregular jewels improve exact
perceptual fidelity without restoring the lattice or violating the sparse/time-tilted structure
gates?

## Source and shared controls

- Architecture: `factorized_structural_jewel_encoder_v3`, 40,960 proposals (`16 x 16 x 8 x 20`).
- Source checkpoint: `irregular_encoder_v2/size_screen/size0023_offset035_seed0_600`.
- Source state: sampled held-out PSNR `18.1175 dB`, median extent `0.03995`, occupancy `0.97593`,
  active fraction `0.66966`, mixed tilt `0.52350`.
- Data: the same 120 training videos, 36 source-owned fitted training fields, and five held-out
  evaluation styles used by the v3 gate.
- Compute: seed 0, 600 continuation steps, AdamW peak LR `1e-4`, warmup 100, 1,024 sampled render
  points/step, 1,024 opacity-sampled student correspondence points, 4,000 teacher points,
  support-tiled renderer capacity 8,192.
- Existing structure losses remain unchanged: Chamfer `0.2`, spread `0.02`, absolute size `0.023`
  with log offset `+0.35`, orientation `0.1`, direct tilt `1.0`, density `2.0`, sparsity `0.1`,
  polarization `0.01`, target active fraction `0.58`.
- Local correspondence: four nearest fitted jewels, Gaussian distance temperature `0.08`, weights
  detached from student centers. Local losses start after step 100 and reach full weight at 400.

## Registered arms

Gradient norms were measured once before registration on the source checkpoint and one training
field, without an optimizer step. Unweighted local/render norms were: scale geometry `2.267`, axis
`0.140`, opacity `0.574`, color appearance `0.187`, gradient appearance `0.0298`, versus render
geometry/appearance `0.101/0.0270`. The following weights keep the new aggregate near the existing
gradient scale.

| Arm | Local scale | Local axis | Local opacity mass | Local RGB | Local RGB gradient |
|---|---:|---:|---:|---:|---:|
| control | 0 | 0 | 0 | 0 | 0 |
| appearance-only | 0 | 0 | 0 | 0.10 | 0.20 |
| full-local | 0.01 | 0.05 | 0.02 | 0.10 | 0.20 |

Opacity is compared in optical-density space and multiplied by full teacher active count divided
by the declared student target active count. Scale compares ordered log-eigenvalues with the same
`+0.35` coverage offset as the successful absolute-size screen. Axis comparison is sign-invariant.

## Screen and exact decision

An intervention remains eligible only if sampled held-out evaluation has occupancy `<=0.985`,
active fraction `<=0.70`, mixed tilt `>=0.25`, and PSNR no more than `0.50 dB` below the matched
control. Every eligible intervention is exact-audited beside the control under the same process.

The local-distillation hypothesis passes only if one arm improves **both** exact PSNR and LPIPS
over control while retaining all three structure gates. Promotion still requires absolute exact
PSNR `>=20 dB` and LPIPS `<=0.40`; only a promoted arm is replicated at seeds 1 and 2. A failed arm
rejects only these weights, kernel, source checkpoint, and 600-step schedule.

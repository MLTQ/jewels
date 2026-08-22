# Local fitted-teacher distillation v1

## Outcome

This experiment found a real but incomplete appearance signal. Matching nearby fitted-teacher
attributes improved exact LPIPS on held-out videos while keeping the learned field irregular,
sparse, and strongly tilted through spacetime. It did **not** improve exact PSNR, did not recover
object boundaries visibly, and therefore did not pass the registered joint gate or the absolute
`20 dB / 0.40 LPIPS` promotion gate.

The negative claim is deliberately narrow: one seed, 600 continuation steps, four-neighbor
position-only correspondence, and the registered loss weights. The positive result is also narrow
but repeatable across the five held-out styles within this run: the full-local arm improved LPIPS
in all five.

## Intervention

`sol/local_teacher_distillation.py` extracts ordered covariance scales, principal axes, opacity,
RGB, and RGB gradients from support-correct fitted fields. A detached four-neighbor Gaussian
kernel assigns those attributes to a bounded sample of student jewels. Scale, axis, optical mass,
RGB, and RGB-gradient losses remain separately weighted and logged. The factorized-v3 trainer
records the entire local kernel and schedule in checkpoint metadata.

The protocol was frozen in `PROTOCOL.md` before optimization. All arms continue the same
40,960-proposal factorized-v3 checkpoint for 600 steps with identical data, seed, optimizer,
renderer, and existing structural losses.

## Registered screen

| Arm | Sampled PSNR | Delta vs control | Occupancy | Active | Mixed tilt | Eligible |
|---|---:|---:|---:|---:|---:|:---:|
| control | 18.3903 | — | 0.97978 | 0.66217 | 0.52453 | reference |
| appearance-local | 18.2318 | -0.1584 | 0.98011 | 0.66650 | 0.52435 | yes |
| full-local | 18.2059 | -0.1844 | 0.98314 | 0.69466 | 0.52249 | yes |

Both interventions remain within the preregistered `0.50 dB` screen tolerance and pass occupancy
`<=0.985`, active fraction `<=0.70`, and mixed tilt `>=0.25`. Full-local comes close to the active
limit, so higher local optical/scale pressure should not be assumed safe.

## Exact-render audit

| Arm | PSNR | Delta | LPIPS | Improvement | SSIM | Joint win? |
|---|---:|---:|---:|---:|---:|:---:|
| control | 18.1654 | — | 0.78204 | — | 0.71685 | reference |
| appearance-local | 18.0917 | -0.0738 | 0.77544 | 0.84% | 0.71629 | no |
| full-local | 18.0399 | -0.1255 | 0.75963 | 2.87% | 0.70880 | no |

Full-local improves LPIPS in all five styles, by `0.0139` to `0.0357`, but lowers exact PSNR in
all five, by `0.0598` to `0.2227 dB`. Appearance-local improves LPIPS in four of five styles and
is a joint PSNR/LPIPS win only for render3d, not at the five-style macro level. This fails the
registered requirement that one macro arm improve both metrics.

The contact sheet shows the same limitation without metric interpretation: all irregular arms
retain broad, washed-out forms and missing object boundaries. The full-local render is sometimes
slightly more differentiated, consistent with LPIPS, but macro-layout is not restored. The field
layout image confirms that these are not lattice renders: all three candidate center fields are
visibly irregular in both XY and XT slices.

## Post-distillation relaxation

Because the full-local LPIPS gain covered all five styles, `PROTOCOL_RELAXATION.md` registered a
second bounded test before running it. All local losses are removed, then the full-local checkpoint
and its matched control receive the same additional 600 render/structure steps.

| Arm at total step 1,200 | Sampled PSNR | Occupancy | Active | Mixed tilt | Eligible |
|---|---:|---:|---:|---:|:---:|
| relaxed control | 18.5603 | 0.98202 | 0.64384 | 0.52159 | reference |
| relaxed full-local | 18.4762 | 0.98267 | 0.63700 | 0.52133 | yes |

The sampled gap contracts from `-0.1844` to `-0.0842 dB`, and the full-local arm moves safely away
from the active-fraction limit.

| Arm at total step 1,200 | Exact PSNR | Delta | Exact LPIPS | Improvement | SSIM | Joint win? |
|---|---:|---:|---:|---:|---:|:---:|
| relaxed control | 18.3653 | — | 0.76791 | — | 0.73709 | reference |
| relaxed full-local | 18.2892 | -0.0761 | 0.76133 | 0.86% | 0.73164 | no |

The distilled arm retains a smaller LPIPS advantage in all five styles (`0.00041` to `0.01358`),
but loses exact PSNR in all five (`0.0137` to `0.1709 dB`). Relative to the first full-local audit,
relaxation recovers `0.2493 dB` PSNR while LPIPS worsens by only `0.00171`; however, the equally
continued control improves enough to remain ahead in PSNR. More of the same relaxation is therefore
not promoted from this checkpoint.

## Evidence

- `evidence.png`: sampled, exact, structural, and per-style registered-arm graph.
- `audit_control_appearance_full_seed0_600/qualitative.png`: target/lattice/candidate/teacher
  exact-render contact sheet.
- `audit_control_appearance_full_seed0_600/field_layout.png`: XY and XT active-center slices.
- `audit_control_appearance_full_seed0_600/comparison.png`: exact macro metric and structure gates.
- `audit_control_appearance_full_seed0_600/report.json`: source-owned exact records.
- `relaxation/audit_control_vs_full_relaxed_total1200/qualitative.png`: relaxed exact-render
  contact sheet.
- `relaxation/audit_control_vs_full_relaxed_total1200/field_layout.png`: relaxed XY/XT center
  slices.
- `relaxation/audit_control_vs_full_relaxed_total1200/comparison.png`: relaxed exact metric graph.
- `relaxation/audit_control_vs_full_relaxed_total1200/report.json`: relaxed source-owned records.

## Interpretation and next gate

The experiment supports two conclusions and no broader law:

1. The v3 irregular field is not quantized to its proposal grid after training; occupancy and direct
   center plots both falsify that concern for these checkpoints.
2. Local fitted-teacher attributes contain a usable perceptual signal, but position-only raw-jewel
   matching spends pixel fidelity for it. Gaussian parameters are compositional: several overlapping
   jewels can render the same observation, so an individual teacher jewel's RGB/covariance is not a
   uniquely identifiable target for the nearest student center.

Equal relaxation did not produce a joint exact win. The next experiment should replace raw
nearest-attribute targets with **renderer-mediated responsibility targets**: weight teacher
attributes by their opacity and covariance support at each student/query location, supervise
composited local moments rather than one teacher jewel, and retain the existing detached-center,
matched-control, structural, exact-audit, and visual gates.

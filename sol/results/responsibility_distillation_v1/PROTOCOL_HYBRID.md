# Hybrid responsibility/local appearance v1 — frozen protocol

## Question

Can the PSNR/layout gain from renderer-weighted appearance responsibility and the LPIPS gain from
position-only local appearance be combined into the first exact joint improvement over an irregular,
sparse, time-tilted matched control?

## Evidence available before registration

- Position-only appearance at RGB `0.10` / gradient `0.20` improved exact LPIPS by `0.00660`
  (`0.84%`) but lost `0.07377 dB` PSNR under its earlier matched source and schedule.
- Responsibility appearance at RGB `0.025` / Jacobian `0.025` improved exact PSNR by `0.05865 dB`,
  SSIM by `0.00372`, and layout PSNR by `0.08994 dB`; LPIPS was effectively tied but `0.00018`
  worse. All structural gates passed.
- These results came from different continuation stages, so their effects are not assumed linear.
  They motivate one bounded combination; they do not predict its outcome.

## Registered arm and control

- Source, seed, data, optimizer, 600-step compute, global structure losses, render sampling,
  responsibility teacher sampling/support, and five validation styles are identical to
  `PROTOCOL.md`.
- Matched control: the already completed `screens/control_seed0_600` arm. Independent teacher RNGs
  ensure the responsibility and position-only samples do not perturb the control's standard teacher
  descriptors or GPU training sequence.
- Hybrid arm: responsibility RGB `0.025`, responsibility Jacobian `0.025`, position-only local RGB
  `0.05`, and position-only local RGB gradient `0.10`. All local/responsibility geometry and opacity
  weights are zero.
- Both objectives start after step 100 and linearly reach full weight at step 400. Position-only
  correspondence retains four neighbors and temperature `0.08`; responsibility retains active-
  uniform 4,000-teacher sampling, five-sigma support, and temperature `1.0`.

The raw local appearance weights are exactly half the earlier appearance-only intervention. This is
strong enough to test the measured LPIPS direction while leaving the responsibility arm's measured
PSNR margin room to survive; it is not chosen from hybrid optimizer results.

## Decision rule

The hybrid remains eligible only if sampled occupancy is `<=0.985`, active fraction is `<=0.70`,
mixed tilt is `>=0.25`, and PSNR is no more than `0.50 dB` below the same matched control. If
eligible, it is exact-audited beside that control.

The hybrid hypothesis passes only if exact macro PSNR is greater and LPIPS is lower than control,
with all structural gates retained. Promotion still requires absolute exact PSNR `>=20 dB` and
LPIPS `<=0.40`; only promotion triggers unchanged seed replication. Failure rejects only this
half-strength appearance combination under the declared source and schedule.

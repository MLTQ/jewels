# Factorized v3 capacity screen — frozen protocol

## Question

Does separating appearance parameters from geometry and increasing the irregular proposal budget
move the encoder into the joint high-fidelity/low-uniformity quadrant?

## Matched screen

- Architecture: `factorized_structural_jewel_encoder_v3`.
- Geometry: mobile stratified centres, quaternion/log-scale covariance, direct mixed-spacetime-tilt
  supervision, opacity-weighted density matching, and explicit sparsity.
- Appearance: independent 32-channel fine/coarse feature volumes, 64-channel per-jewel MLP,
  continuous RGB seed, and bounded colour gradient. Appearance sampling uses detached centres.
- Capacity arms: 10, 20, and 36 proposals per `16 x 16 x 8` cell: 20,480, 40,960, and 73,728
  proposals per video window.
- Data: all 120 training clips receive render supervision; the same 36 source-owned fitted fields
  receive geometry supervision; five disjoint validation styles remain evaluation-only.
- Compute: seed 0, 600 steps, 1,024 sampled render points/step, support-tiled five-sigma renderer,
  capacity 8,192, identical optimizer and schedules.
- Structural schedule: sparsity, density, orientation, and direct tilt begin at step 100 and reach
  full weight at step 400. Active target is 0.58.

## Screen selection

An arm is eligible for continuation only if its sampled held-out evaluation has:

1. occupancy uniformity `<= 0.985`;
2. active proposal fraction `<= 0.70`;
3. mixed-spacetime tilt median `>= 0.25`;
4. no support-capacity overflow or teacher/source mismatch.

Among eligible arms, continue the highest-PSNR capacity. If none qualifies, report the closest
tradeoff and revise the declared architecture or schedule; do not reinterpret thresholds.

## Promotion

The continued seed-0 arm still must pass the original exact audit: PSNR `>= 20 dB`, LPIPS `<= 0.40`,
occupancy `<= 0.985`, active fraction `<= 0.70`, and mixed tilt `>= 0.25`. Only then run seeds 1 and
2. A limited failure applies only to its capacity and training configuration.

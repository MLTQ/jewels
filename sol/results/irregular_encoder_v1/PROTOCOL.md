# Irregular encoder gate — preregistered protocol

## Question

Can a feed-forward video encoder retain useful reconstruction while emitting an opacity-sparse,
content-clustered field of mobile anisotropic jewels rather than a fixed RGB sampling grid?

## Frozen comparison

- Baseline: the support-correct `n120/seed0` lattice encoder (73,728 always-active slots).
- Ceiling: five independently fitted, source-owned validation fields, one per visual style.
- Candidate: 20,480 irregular proposals (10 per 16 x 16 x 8 cell), up to four-cell migration,
  continuous colour seeding at predicted centres, quaternion/scale covariance, and support-complete
  rendering.
- Training: all 120 training clips receive render supervision; the preregistered 12-clip balanced
  subset additionally receives source-owned fitted-field structure supervision. Validation teachers
  are never used for training.
- Replication: select loss weights with bounded seed-0 screens, then train the selected arm at seeds
  0, 1, and 2. A failed screen narrows only that configuration.

## Primary gate

The replicated candidate passes only if the five-source held-out macro satisfies all of:

1. mean occupancy uniformity <= 0.985 and lower than the lattice in every seed;
2. mean active proposal fraction <= 0.70;
3. mean mixed-spacetime tilt median >= 0.25;
4. mean PSNR >= 20 dB and LPIPS <= 0.40;
5. no support-capacity overflow and no source/teacher ownership mismatch.

These thresholds test the user's observed defect directly. They are evidence that more compute can
improve a viable representation, not a claim that this small encoder is already a text-to-video
model.

## Reports

The run must preserve command/config metadata, per-source/per-seed records, a qualitative contact
sheet, and graphs for LPIPS, PSNR, occupancy uniformity, active fraction, and training curves.

## Execution notes (thresholds unchanged)

- The structure-teacher corpus was expanded from the preregistered 12 balanced clips to 36/120
  training clips after the 12-teacher arm showed a positive fidelity/sparsity trajectory but missed
  occupancy. The five validation teachers remained disjoint and evaluation-only.
- The direct mixed-spacetime-tilt loss was selected after the cosine-orientation screen failed to
  change the measured gate quantity. This narrows that surrogate; it is not evidence against tilted
  jewels.
- Seed 0 did not pass the full primary gate, so seeds 1 and 2 were not promoted to expensive
  replicas. No three-seed success is claimed.
- A post-screen causal continuation froze the successful step-2,000 geometry exactly and trained
  appearance rows for 4,000 more steps. It is diagnostic follow-up, not a change to the gate.

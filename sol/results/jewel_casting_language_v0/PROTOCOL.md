# Jewel casting language Gate 0 protocol

## Decision question

Do independently optimized irregular Jewelfields admit a reusable discrete-continuous casting
language, or does decomposition gauge make the token targets arbitrary?

This is not a compression experiment. A cast is counted as a generative decision, and continuous
residuals are deliberately retained so their remaining burden can be measured.

## Frozen representation

- Canonical source features: center 3, log-covariance 6, RGB 3, RGB gradient 9, opacity logit 1.
- Address grid: `8 x 8 x 4`; the address never replaces the continuous center.
- Bundle size: eight canonically ordered, cell-local Jewels per cast; partial bundles preserve an
  explicit count and no target may be truncated.
- Vocabulary budgets: 64, 256, and 1,024 learned joint constellation motifs.
- Vocabulary fitting: source-disjoint optimized irregular fields only; 100k sampled bundles,
  deterministic seed 20260826, 15 Lloyd iterations.
- Candidate decode: motif only, motif plus 50% residual, and full residual.
- Negative control: exact intrinsic attributes with every center moved to its addressed cell center.

## Equivalence audit

Anime cooking, photoreal surfing, and clay campfire are each fit independently under seeds 0, 1,
and 2 using the same 72k-Jewel, 3,000-step support-correct protocol. All three source identities are
excluded from vocabulary fitting. Same-source program similarity is compared with every
different-source pairing; cell-conditional motif histograms prevent occupancy alone from deciding
the result.

## Registered positive gate at vocabulary 1,024

All checks must pass:

1. Every source Jewel is serialized and decoded; mean casts are at most 12,000 per 72k field.
2. Motifs explain at least 25% of standardized local-bundle energy before continuous residuals.
3. The 50%-residual decode reaches at least 30 dB on identical support-rendered random volume
   points and retains median mixed spacetime tilt within 10% of the continuous source.
4. Motif-only centers have less than 1% exact cell-center locking.
5. Same-source cell-conditional motif cosine exceeds different-source cosine by at least 0.05.
6. Full-residual decoding is numerically exact at at least 80 dB.

Failure blocks transformer training. A failed individual vocabulary is a result for that budget,
not a universal law; changing bundle factorization or separating geometry/appearance requires a new
registered protocol rather than retroactive reinterpretation.

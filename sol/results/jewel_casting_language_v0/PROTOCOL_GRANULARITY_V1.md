# Jewel casting granularity Gate 0c protocol

## Decision question

Does Gate 0b fail because physical roles are wrong, or because each role prototype still averages
eight distinct Jewels? This test approaches the original proposal directly: a predefined Jewel
description plus its irregular continuous centroid.

## Frozen comparison

- Reuse Gate 0b without changing its fields, train/validation split, four role definitions,
  K=1,024 per role, fitting iterations, renderer, points, independent seeds, or residual arms.
- Change only canonical bundle size: 8, 4, 2, and 1 Jewels per cast.
- Bundle 1 is the registered primary upper bound. Its exact continuous anchor is the Jewel centroid;
  layout becomes a constant token, while covariance, surface, and gradient remain learned discrete
  descriptions.
- Report complete-field decisions and the equivalent decisions for an eight-frame generation
  window using the frozen 49-frame sources. This is a generation-compute curve, not a compression
  claim.

The existing Gate-0b bundle-8 report is reused. Bundle 4, 2, and 1 are new matched audits. Their
embedded Gate-0b verdicts are ignored because Gate 0b registered a different decision budget;
Gate 0c alone owns the granularity decision.

## Registered positive gate at bundle 1

All checks must pass:

1. Every Jewel is serialized, full-residual rendering is at least 80 dB, and exact center locking is
   below 1%.
2. Token-only rendering reaches at least 20 dB and improves at least 3 dB over the bundle-8
   factorized phrase.
3. The 50%-residual arm reaches at least 25 dB and retains mixed spacetime tilt within 10%.
4. Same-source composite cell-conditional cosine exceeds different-source cosine by at least 0.02.
   Gate 0a/0b already own the stronger coarse semantic-language threshold; micro tokens need only
   remain learnably source-conditional rather than semantically canonical one by one.
5. Four role decisions per Jewel fit at most 50,000 role-token decisions in an eight-frame window.
6. Token-only and half-residual PSNR are monotonically non-decreasing as bundle size shrinks.

A pass proves a usable upper-bound Jewel language and licenses hierarchical coarse-to-micro
generation. It does not prove the micro tier is efficient or promptable; those are the next gates.
A failure localizes the remaining obstacle to finite attribute vocabulary/continuous prediction,
not lattice placement or bundle grouping.

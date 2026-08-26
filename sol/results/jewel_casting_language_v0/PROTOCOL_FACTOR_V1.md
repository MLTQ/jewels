# Factorized Jewel casting language Gate 0b protocol

## Why this is a new registered test

Gate 0a found a large, replicated same-video language margin but rejected one joint token spanning
eight Jewels and all 22 dimensions. That failure is scoped to a 176-dimensional joint prototype:
it does not test a compositional phrase whose tokens own different physical roles.

Gate 0b keeps the corpus, source-disjoint split, cells, canonical bundling, exact continuous
centroids, validation videos, renderer, and independent fit seeds unchanged. It changes only the
motif factorization and declares its gate before results are observed.

## Frozen casting phrase

Every eight-Jewel cast emits one coarse cell address, one exact continuous anchor, one exact count,
and four discrete role tokens:

1. `layout`: eight relative continuous centers (24 dimensions);
2. `covariance`: eight log-covariances, including spacetime orientation (48 dimensions);
3. `surface`: eight RGB-plus-opacity tuples (32 dimensions);
4. `gradient`: eight first-order RGB spacetime gradients (72 dimensions).

Rows remain aligned by the same within-cell lexicographic center order. Each role has an independent
64, 256, or 1,024-entry vocabulary. At 1,024 entries, four 10-bit choices compose a space of up to
`1024^4` cast phrases without fitting an impossibly large joint table. Continuous residuals remain
in the audit so their burden is measured; they are not claimed as compression.

## Frozen data and fitting

- Vocabulary train set: the same 33 source-disjoint, 72k-Jewel optimized irregular fields.
- Validation: anime cooking, clay campfire, and photoreal surfing, each optimized independently at
  seeds 0, 1, and 2.
- Bundle sampling: 100,000 train casts, deterministic seed 20260827, 15 Lloyd iterations.
- Renderer: support-complete random-volume points shared across all arms.
- Candidates: composed role tokens only, tokens plus 50% exact residual, and full residual.
- Negative control: exact intrinsic attributes with all centers moved to cell centers.

## Registered positive gate at four vocabularies of 1,024

All checks must pass:

1. Every Jewel is serialized; mean casts are at most 12,000 and discrete role decisions at most
   40,000 per 72k field.
2. The composed phrase explains at least 35% of standardized bundle energy.
3. Token-only random-volume PSNR improves at least 2 dB over Gate 0a's matched K=1,024 joint token.
4. The 50%-residual decode reaches at least 20 dB, improves at least 3 dB over Gate 0a's matched
   half-residual arm, and retains median mixed spacetime tilt within 15% of the source.
5. Token-only centers have less than 1% exact cell-center locking.
6. Same-source cell-conditional composite-token cosine exceeds different-source cosine by at least
   0.05.
7. Full-residual decoding is numerically exact at at least 80 dB.

A pass licenses residual-prediction and free-running tests. A failure does not license threshold
changes; it distinguishes which role remains too entropic and determines whether individual-Jewel
tokens or learned residual refinement are the next registered representation.

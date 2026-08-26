# Hierarchical Jewel casting language Gate 0d protocol

## Motivation fixed from prior diagnoses

Gate 0c established two complementary facts:

- two-Jewel phrases are source-stable and preserve spacetime covariance but do not reach the
  registered token-only rendering bar;
- individual-Jewel phrases reach 23.15 dB token-only and 27.98 dB at half residual, but their
  constant layout axis invalidates a naïve composite cosine and their covariance over-amplifies
  temporal tilt.

Gate 0d composes only the demonstrated responsibilities: pair-level layout and covariance;
individual-Jewel surface and gradient. It is a new architecture tested on fresh held-out sources,
not a reinterpretation of Gate 0c.

## Frozen fresh validation

Vocabulary training excludes these three sources completely:

- `cartoon__01_dogpark_train_00_seed61100`
- `render3d__04_workshop_train_00_seed62400`
- `anime__05_ballet_train_00_seed60500`

Each is optimized independently at fitter seeds 0, 1, and 2 with the same 72k-Jewel, 3,000-step,
2,048-voxel support-correct protocol. Seed 0 already exists; seeds 1 and 2 are fit after this
protocol is frozen. All remaining available irregular fields train two new source-disjoint K=1,024
factorized codebooks at bundle sizes 2 and 1.
Both use 100,000 sampled casts and 15 Lloyd iterations; deterministic seeds are 20260828 for the
pair codebook and 20260829 for the individual codebook.

## Frozen hierarchical phrase

Per two target Jewels:

1. one coarse cell address, exact continuous pair anchor, exact count;
2. pair-level layout token;
3. pair-level covariance token;
4. one individual surface token per Jewel;
5. one individual color-gradient token per Jewel.

Token-only and 50%-residual candidates take center/covariance dimensions from the pair program and
appearance dimensions from the individual program. Full residual is the exact consistency arm.
The estimated role-token cost is two pair decisions plus two individual decisions per Jewel, or at
most 40,000 decisions for an eight-frame slice of a 49-frame/72k-Jewel field.

## Registered fresh-source gate

All checks must pass over the nine fresh independent fields:

1. Every Jewel is serialized; full-residual rendering is at least 80 dB; token centers have less
   than 1% exact cell-center locking.
2. Token-only random-volume PSNR is at least 20 dB.
3. The 50%-residual arm is at least 25 dB and retains median mixed spacetime tilt within 10%.
4. The cell-conditional histogram formed from pair covariance plus individual surface and gradient
   tokens has same-source-minus-different-source cosine margin at least 0.05.
5. Equivalent eight-frame role-token decisions are at most 40,000.

A pass licenses a free-running prompt-conditioned caster. It is evidence that a stable, irregular,
render-capable Jewel action language exists; it is not yet evidence that text can generate the
language. A failure sends the project to learned continuous heads rather than another hand-selected
token recombination.

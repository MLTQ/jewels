# Gate 2a4 protocol: predefined block-constellation oracle

Frozen after the K=1024 empirical macro-token review and before implementation or execution.

## Evidence motivating this test

The K=1024 empirical macro-token passes every numerical Gate 2a threshold: direct token NLL is
5.803 (5.2% better than the global posterior), direct histogram cosine is 0.548 (+0.260), oracle
beats shuffled/null on direct and source-disjoint data, and centroid grid locking is zero. Yet its
renders remain textured fields rather than recognizable subjects.

The empirical realizer pools all Jewel tuples assigned to a token and samples them independently.
It therefore proves that local token distributions exist while intentionally discarding correlations
between the Jewels in one block. The user's literal proposal is stronger: a token maps to predefined
Jewels **and their centroid positions as one joint local utterance**. This gate tests that claim.

## Frozen constellation definition

- Reuse the immutable K=1024 block vocabulary and 18 training block programs.
- For every utilized block token, choose exactly one training block occurrence: the medoid whose
  normalized 77D descriptor has minimum squared distance to that token's frozen prototype.
- Store that occurrence's complete ordered set of
  `(continuous block-local centroid, covariance token, surface token, gradient token)` tuples.
- A token occurrence casts the entire stored constellation into its addressed block. Add Gaussian
  jitter with standard deviation 0.005 in block-local coordinates, then clamp only to the open
  block boundary.
- Concatenate all 256 constellations. If their total differs from 72,000, perform one declared global
  random subsample or jittered bootstrap to emit exactly 72,000 Jewels. Report the unadjusted count
  and adjustment fraction.
- Use additive smoothing 0.1 over each medoid constellation's role histograms for token NLL.

## Data, controls, and gate

- Medoids come only from the 18 exact-prompt training fields.
- Evaluate the same direct and source-disjoint fields with target-derived oracle programs,
  cyclic-shuffled programs, and the most frequent nonempty null token.
- Use matched random seeds and unchanged global-posterior thresholds.
- Require all numerical Gate 2a checks, zero/under-1% grid locking, and finite renders.
- Require at least two of three source-disjoint oracle rows to show a localized subject or coherent
  scene structure that is absent from shuffled and null rows.

If this passes, a finite token-to-joint-constellation language exists and Gate 2b may learn to emit
its ordered tokens from text. If numerical metrics pass but recognizable structure still fails,
8x8x4 blocks are too coarse or independently assembled; the next architecture must add a coarser
object/region level above block tokens rather than merely train longer.

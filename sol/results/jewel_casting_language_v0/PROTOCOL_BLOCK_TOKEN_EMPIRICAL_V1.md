# Gate 2a2 protocol: empirical macro-Jewel block oracle

Frozen after the local-only neural K=64/256/1024 curve and before this implementation or run.

## Why this is a distinct test

Gate 2a showed that target-derived local block tokens strongly control structure but a neural
conditionally independent point sampler turns that structure back into texture. At K=256 the
target-histogram cosine rose from the global oracle's 0.288 to 0.415, correct blocks beat shuffled
and null blocks on seen and source-disjoint fields, and grid locking remained zero; however token
NLL missed the frozen global baseline by 0.2% and no rendered subject was recognizable.

That result falsifies **replacing** the global scene with an independent local point sampler. It
does not test the user's proposed token semantics: a token maps to a predefined local collection of
Jewels and relative centroid positions. Gate 2a2 tests that semantics directly without changing the
frozen K=256 block vocabulary or looking at held-out assignments.

## Fixed empirical macro token

- Reuse the exact K=256 training-owned block codebook and all 18 training programs from Gate 2a.
- For every block token, collect a source-pooled empirical reservoir of tuples
  `(continuous local centroid, covariance token, surface token, gradient token)` from all training
  block occurrences assigned that token.
- Store a mean emitted-Jewel count per block-token occurrence and smoothed per-role token
  histograms. Use additive smoothing 0.1 for reported token NLL.
- A block-token occurrence samples a block count, then samples complete tuples from its predefined
  reservoir. Add independent Gaussian jitter with standard deviation 0.01 in block-local
  coordinates and clamp only to the open block boundary. The output is therefore continuous and
  cannot lock to grid or microcell centers.
- Allocate exactly 72,000 Jewels across the 256 block occurrences in proportion to the frozen mean
  count for each emitted block token. Randomness is fully declared and matched across arms.

No neural decoder is trained in this gate. The empirical macro token is a finite constructive
existence proof: it asks whether a reusable token-to-local-Jewel mapping can carry object structure.

## Data and controls

- Fit reservoirs only from the 18 exact-prompt training sources.
- Direct comparison uses the same first training source per prompt as the global posterior oracle.
- Source-disjoint evaluation uses all three independent fits of the held-out original sources for
  likelihood and the lowest fit seed for renders.
- Arms are target-derived oracle blocks, cyclic-shuffled blocks, and a prompt-blind program made
  from the most frequent nonempty training block token.
- Import the frozen global posterior/text/null baselines without retraining.

## Gate

Advance if all Gate 2a thresholds hold unchanged:

- direct token NLL improves by at least 2% over global posterior NLL 6.123414781358508;
- direct target-histogram cosine improves by at least 0.02 over 0.28816187381744385;
- oracle beats shuffled and null on both token NLL and histogram cosine in direct and
  source-disjoint evaluations;
- centroid locking is below 1% and all renders are finite.

Additionally, qualitative inspection must show a spatially localized subject/layout rather than
the globally washed texture seen in Gate 2a. Recognizable semantic identity is supportive here but
remains mandatory only after text emits the block program in Gate 2b.

Passing proves a finite native macro-Jewel token language exists. It still does not prove text
promptability; only then may Gate 2b train an autoregressive prompt-to-block sequence model.

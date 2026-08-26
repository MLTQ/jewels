# Gate 2a3 protocol: empirical macro-token capacity upper bound

Frozen after reviewing the K=256 empirical macro-token result and before this run.

## Motivation

K=256 empirical realization preserves a causal local-program signal but still averages recognizable
subjects into texture: direct/source-disjoint histogram cosine is 0.426/0.267, both beat shuffled
and null, and grid locking is zero, while direct NLL misses the global posterior by 0.06%.

The original Gate 2a protocol preregistered K=1024 as the capacity upper bound. At K=1024 each
token is supported by about four training block occurrences rather than about eighteen at K=256.
This test determines whether macro-token averaging, rather than block granularity itself, caused the
qualitative failure.

## Frozen experiment

- Reuse the completed K=1024 neural Gate 2a checkpoint only for its immutable training-owned block
  codebook and 18 block programs.
- Fit the same empirical tuple reservoirs as Gate 2a2: smoothing 0.1, block-local jitter 0.01,
  exactly 72,000 emitted Jewels, identical direct/source-disjoint arms and matched randomness.
- Do not alter the descriptor, 8x8x4 block geometry, active K=1024 Jewel language, data split,
  baseline values, or rendering settings.

## Decision

Apply the same numerical thresholds as Gate 2a2. In addition, at least two of three source-disjoint
oracle renders must show a spatially localized subject or scene structure that is absent under the
shuffled and null programs.

If K=1024 passes, use K=1024 for the first prompt-to-block speaker. If it improves K=256 but remains
unrecognizable, the next test must increase spatial hierarchy or emit explicit local constellations;
do not claim that more training of the same 8x8x4 averaged-token formulation will solve it.

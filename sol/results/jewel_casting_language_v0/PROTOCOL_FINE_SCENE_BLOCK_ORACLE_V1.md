# Gate 2a6 protocol: fine scene/block constellation oracle

Frozen after the 8x8x4 scene/block qualitative review and before implementation or execution.

## Question

The 8x8x4 two-level hierarchy passes every numerical control but does not resolve recognizable
subjects. Is the remaining loss caused by routing granularity rather than by the scene/block token
syntax itself?

## Fixed change

- Keep the Gate 2a5 hierarchy, data, exact prompt-owned scene tokens, medoid rule, four-neighbor
  likelihood pool, smoothing 0.1, jitter 0.005, exact-72k adjustment, controls, and render settings.
- Change only the internal routing shape from 8x8x4 (256 blocks) to **16x16x8 (2,048 blocks)**.
- Fit a fresh K=1024 block descriptor vocabulary on all 36,864 fine training blocks using the same
  77D descriptor and 20 deterministic Lloyd iterations.
- Serialize the future program time-major with Morton/Z spatial order. Fine routing addresses never
  replace emitted continuous centroids.
- Keep evaluation histograms and the global-posterior comparison on the original 8x8x4 grid so the
  frozen numerical thresholds remain directly comparable. Separately report locking under the fine
  routing grid.

The resulting native utterance has one scene decision plus 2,048 block decisions for a 49-frame
window, versus about 216,000 individual Gate 0f Jewel-role decisions. This is a structural language,
not a codec claim.

## Gate

- Direct token NLL improves at least 2% and fixed-grid histogram cosine at least 0.02 over the global
  posterior.
- Oracle hierarchy beats shuffled-scene, shuffled-block, and null on token NLL and fixed-grid
  histogram cosine, direct and source-disjoint.
- Both 8x8x4 evaluation-grid and 16x16x8 routing-grid center locking remain below 1%; renders finite.
- At least two of three source-disjoint oracle rows show recognizable/localized prompt-consistent
  subject or scene structure absent from every control.

If this passes, freeze 16x16x8/K1024 and proceed to prompt-to-program learning. If it again passes
metrics but fails qualitatively, independent spatial blocks are insufficient even at fine scale;
add object/track tokens above them rather than increasing training duration.

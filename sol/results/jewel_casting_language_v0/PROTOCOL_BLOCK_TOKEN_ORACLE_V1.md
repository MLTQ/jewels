# Gate 2a protocol: local spacetime block-token oracle

Frozen before implementing or training the Gate 2a model.

## Question

Does one shared discrete token per local spacetime block supply the missing structural state that a
single global scene vector could not? This is a deliberately target-leaking oracle diagnostic. It
does not count as prompt-to-video generation.

## Fixed representation

- Keep the proven Gate 0f output language: exact continuous centroids plus independent K=1024
  covariance, surface, and gradient Jewel tokens.
- Partition normalized `(x, y, t)` only for routing into `8 x 8 x 4 = 256` blocks. Routing never
  replaces or snaps a generated centroid.
- Serialize blocks time-major. Within each time slab, serialize `(x, y)` in Morton/Z order.
- Describe each block with 77 target-derived values: log occupancy; a normalized `4 x 4 x 2`
  local occupancy histogram; local centroid mean and standard deviation; and mean and standard
  deviation of the 19 normalized intrinsic Jewel features.
- Standardize descriptors using the 18 registered exact-prompt training sources only. Fit a
  deterministic Lloyd vocabulary to all training blocks, including empty blocks.
- Primary vocabulary size: 256. Frozen capacity checks: 64 and 1024. The 256 result owns the gate;
  the other sizes prevent one under-capacity result from being reported as a general law.

The block token is the only target-derived decoder condition. The decoder also receives the fixed
block address and each proposed continuous coordinate relative to its block. It never receives
pixels, a dense video latent, a source identifier, or the target Jewel row at generation time.

## Data

- Train the block vocabulary and Jewel expander on the 18 exact-prompt sources used by Gate 1h:
  six independent LTX videos for each of ballet/anime, dog-in-park/cartoon, and
  welder/workshop/3D-render.
- Direct-comparison oracle set: the same lexicographically first training source per prompt used by
  the frozen global-posterior oracle report.
- Source-disjoint transfer set: the three held-out original sources, retaining all three independent
  Jewelfield fits for teacher-forced metrics and the lowest fit seed for rendering.
- Sample at most 16,384 positive Jewels per training source. Train for at most 20,000 steps,
  evaluate every 500 steps, and stop only after 12 evaluations without a 0.1% validation-score
  improvement. Select the checkpoint on source-disjoint oracle token NLL plus `0.1 * density NCE`.
- Generate exactly 72,000 Jewels per program from four-times as many uniform continuous proposals,
  using temperature 0.9 and top-k 64 for each active Jewel role.

## Arms

For each evaluated target, use identical random proposals and token-sampling seeds.

1. **Oracle block**: target-derived block descriptors assigned to the frozen block vocabulary.
2. **Shuffled block**: the oracle block program from the next prompt class.
3. **Null block**: the most frequent training block token in every address.
4. Import the frozen shared-scene **global posterior oracle**, **correct text**, and **prompt blind**
   results as named baselines; do not retrain or reinterpret them.

## Primary gate

The primary K=256 oracle advances only if all of the following hold:

- Direct-comparison token NLL improves by at least 2% over the frozen global posterior oracle
  (`6.123414781358508`).
- Direct-comparison target-histogram cosine improves by at least 0.02 over the frozen global
  posterior oracle (`0.28816187381744385`).
- Oracle block beats shuffled block and null block on both token NLL and target-histogram cosine.
- Source-disjoint oracle block beats its shuffled and null controls on those same two metrics.
- Generated centroid cell-center locking remains below 1% and all renders are finite.

If K=256 narrowly fails but K=64 and K=1024 show an ordered, reproducible improvement and K=1024
passes every threshold, record a capacity-dependent support result rather than a general failure;
the prompt-to-block stage must then use the smallest passing vocabulary. Otherwise the proposed
local block-token hierarchy is falsified at this granularity.

## Advancement rule

Only a passing local oracle authorizes Gate 2b: predict the ordered block-token sequence from text
and prior block tokens. Gate 2a success proves that a compact hierarchical Jewel utterance can
carry local structure; it does not prove promptability. Gate 2b must retain correct/shuffled/null
controls, replicated seeds, recognizable held-out subject/action, continuous centroids, and later
an overlap/carry test into a second window.

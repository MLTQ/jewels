# Gate 2a5 protocol: scene-token plus block-constellation hierarchy

Frozen after Gate 2a4 and before implementation or execution.

## Causal question

Local K=1024 tokens contain enough information to pass every numerical oracle gate when their
Jewel tuples are pooled, but complete medoid constellations selected independently across blocks
remain stylistically and geometrically incoherent. Does one coarser scene token make local
constellation realization coherent by selecting all block templates from the same semantic family?

## Fixed hierarchy

1. Emit one scene token for the window. For this diagnostic, the three tokens are the three exact
   registered `(style, action prompt)` groups; a fourth token is a prompt-blind pooled control.
2. Emit the existing ordered 256-token K=1024 local block program.
3. The pair `(scene token, block token)` maps to a predefined complete medoid constellation.
4. Cast the medoid's continuous local centroids and active Jewel roles with jitter 0.005, then
   perform the same explicit exact-72k global count adjustment as Gate 2a4.

For each of the three semantic scene tokens and each K=1024 block token, choose the minimum-distance
77D block descriptor among only that scene token's six training videos. The prompt-blind scene
token chooses from all 18 videos. For likelihood, pool the four nearest eligible training block
occurrences per `(scene, block)` pair with additive smoothing 0.1; generation always uses the single
nearest complete constellation so joint geometry is preserved.

The scene token is provided by the registered text label in this oracle. Local block tokens remain
target-derived. This is not prompt-to-video generation; it tests whether the exact two-level token
syntax has sufficient constructive capacity before learning either level.

## Controls

Use identical randomness for:

- **oracle hierarchy**: correct scene token plus target-derived block program;
- **shuffled scene**: next semantic scene token plus the same block program;
- **shuffled blocks**: correct scene token plus the next prompt's block program;
- **null hierarchy**: prompt-blind scene token plus the most frequent nonempty block token at every
  address.

Evaluate the same three direct training sources and all nine source-disjoint fits; render the lowest
fit seed per held-out source. Import the frozen global posterior metrics without retraining.

## Gate

- Direct token NLL improves at least 2% and histogram cosine at least 0.02 over the global posterior.
- Oracle hierarchy beats all three controls on token NLL and histogram cosine, both direct and
  source-disjoint.
- Grid locking stays below 1%; all renders are finite; exact-count adjustment is reported.
- At least two of three source-disjoint oracle rows show a localized, semantically class-consistent
  subject or scene structure absent from shuffled-scene, shuffled-block, and null rows.

If this passes, freeze this two-level vocabulary and train Gate 2b to emit scene then block tokens
from text. If it fails qualitatively, the next necessary level is explicit object/region tracks or
finer spatial blocks; longer training of an independent block assembler is rejected.

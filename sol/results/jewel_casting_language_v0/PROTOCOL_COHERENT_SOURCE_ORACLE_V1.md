# Gate 2a8 protocol: coherent source-program oracle

Frozen after Gate 2a7 passed every numerical structure check but failed the recognizable-subject
review.

## Decision question

Is the qualitative failure caused primarily by choosing local constellations independently, or is
even one complete source-level Jewel program insufficient once reduced to the active K=1,024 Jewel
vocabulary?

This is a causal upper-bound control, not a proposed inference architecture. A pass licenses
object/trajectory tokens; it does not license retrieval as text-to-video.

## Frozen experiment

- Reuse the immutable 18/9 exact-prompt source split, the Gate 0f K=1,024 active Jewel vocabulary,
  the 16x16x8 K=1,024 block language, and the same three source-disjoint qualitative rows.
- Quantize every training field to continuous centroids plus the three active Jewel tokens. No
  continuous appearance residual is retained.
- For each target-derived oracle block program, choose **one** eligible training field for the
  entire window by minimum mean block-prototype distance over all 2,048 fixed addresses.
- The correct arm restricts that one field to the registered semantic scene. The shuffled-scene arm
  restricts it to the next scene. The shuffled-block arm applies the same within-window permutation
  used by Gate 2a7 before choosing one correct-scene field. The null arm chooses from all scenes
  using the most-frequent block token at every address.
- Cast the chosen field's complete quantized Jewel program with continuous centroid jitter 0.005.
  Preserve exactly 72,000 Jewels and use matched random seeds for every arm.

The direct-comparison subset may select itself and is reported only as the finite-vocabulary
ceiling. The source-disjoint rows can select only among the six exact-prompt training sources and
are the causal decision set.

## Gate

1. Source-disjoint correct-scene programs beat shuffled-scene and null controls on active-token
   likelihood and cell-conditioned histogram cosine.
2. Every generated field is finite, has exactly 72,000 Jewels, and has less than 1% exact grid
   locking.
3. At least two of the three source-disjoint correct-scene rows show a recognizable,
   prompt-consistent subject/action that is absent from shuffled-scene and null rows.
4. The report must label the arm as target-program-selected retrieval and must not claim prompt-only
   inference or novel generation.

## Decision rule

- **Pass:** cross-block source coherence is a sufficient missing variable. Implement a compositional
  object/trajectory program in which one persistent token owns a connected spacetime tube, then
  require novelty by combining independently selected foreground and background tubes.
- **Fail:** whole-window coherence does not restore recognizable structure. Stop expanding the
  token hierarchy and replace the active-token realization vocabulary or renderer contract.

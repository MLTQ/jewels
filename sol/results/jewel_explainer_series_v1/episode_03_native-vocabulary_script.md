# Episode 3: Giving the model a Jewel vocabulary

Reusable words for shape and appearance

## Claim sources

- `stprim/prior/featurize.py`
- `sol/factorized_jewel_casting_language.py`
- `sol/prompt_jewel_caster.py`
- `sol/results/jewel_casting_language_v0/hierarchical_v1/gate0f_individual/report.json`

## 1. Why make tokens at all?

A language model works with reusable choices called tokens: words or word-like symbols from a fixed vocabulary. We want similar Jewel shapes and colors to reuse the same tokens across many videos. This is not an attempt to compress a finished video. The goal is to give a future generator a stable set of physical words it can speak into an empty spacetime volume.

**On screen:** Tokens are reusable physical choices, not a video compression format.

## 2. One shape can have many spellings

The same tilted ellipsoid can be written with several different rotation-number combinations. The picture does not change, but the numbers do. A learner would mistake those spellings for different shapes and waste vocabulary space. Before creating tokens, we convert every Jewel to one consistent numerical spelling. In mathematics this removes a gauge ambiguity; in plain language, it removes meaningless aliases.

**On screen:** First give every visible shape one consistent numerical spelling.

## 3. A stable six-number shape

We store each shape as a symmetric three-by-three table, which needs six unique numbers. We also take a matrix logarithm, a standard transformation that makes large and small scales easier to compare. The important result is simple: identical visible shapes now receive identical stored features. The later renderer can convert that stable spelling back into widths and tilt.

**On screen:** A unique six-number shape description replaces ambiguous rotations.

## 4. Separate the physical choices

Our first attempt bundled layout, shape, color, and color change into one enormous choice. That was like printing every possible sentence as a single dictionary entry. It failed. The stronger design gives each role its own list of prototypes. A prototype is simply a learned representative example. The model can then combine a familiar shape with a familiar color in a new way.

**On screen:** Choose layout, shape, color, and color change separately.

## 5. Mix-and-match creates range

Four lists of one thousand twenty-four choices can describe far more combinations than one list of the same size. We do not build a table containing every combination; we compose the roles when needed. For an individual Jewel, position remains continuous while shape, color, and color change come from their learned vocabularies. This gives the speaker reusable parts without forcing every result to copy a stored whole.

**On screen:** A few reusable lists can be mixed into many Jewel combinations.

## 6. A map is not a parking space

The learner uses coarse boxes to ask which tokens are common in each neighborhood. Those boxes are addresses for routing information, like postal codes. They are not the final Jewel positions, just as a postal code is not a chair inside a house. The model still emits a continuous center that can land anywhere. Discrete routing and irregular geometry can safely coexist.

**On screen:** A routing box selects context; it does not snap the Jewel's center.

## 7. The vocabulary keeps the signal

We tested whether replacing continuous features with vocabulary choices destroyed the field. The reconstructed images reached about twenty-two point nine decibels of peak signal-to-noise ratio, a standard image-similarity measure where higher is better. Spacetime tilt was preserved, and no centers locked to the routing grid. This does not prove promptable generation. It shows the physical vocabulary is usable enough for the next level.

**On screen:** Useful image fidelity, preserved motion tilt, and zero center locking.

# Gate 2a10 protocol: semantic density-balanced trajectory oracle

Frozen after Gate 2a9 preserved recognizable subjects in all three composites but failed its
numeric gate: the donor-disagreement tube required 15.54% count adjustment and its low-level
histogram preferred the visibly wrong-object control.

## Correction under test

Gate 2a9 found a region where two donors differed; it did not find the region that defines the
prompt. Gate 2a10 replaces that path with a training-only class-discriminative trajectory and makes
the foreground/background boundary density balanced. It does not change the Jewel vocabulary,
source split, target-program donor selection, renderer, evaluation grid, or qualitative rows.

## Frozen experiment

- For each of the three semantic scene tokens, average normalized 16x16x8 block descriptors over
  its six training fields and over the other twelve fields. Per-address squared mean difference,
  multiplied by the same centered Gaussian prior, defines semantic saliency.
- Compute one saliency-weighted XY centroid per time slab and smooth the path with `[1,2,1]`.
  The path is a function only of the scene token and training fields.
- Select the nearest and next-nearest same-scene donor programs exactly as in Gate 2a9.
- Select a tube radius from 132 fixed candidates spanning 0.45 through 1.10. Among candidates where
  both donors contribute at least 20% of the unadjusted composite, minimize absolute count mismatch;
  ties prefer the radius nearest 0.78. This uses donor density, not target rows or render metrics.
- Cast foreground-donor Jewels inside the semantic tube and background-donor Jewels outside it,
  then uniformly adjust to exactly 72,000 rows and jitter by 0.005.
- Controls remain the coherent ceiling, wrong-scene foreground inserted through the **correct
  scene's** semantic tube, and pooled-null coherent source.

## Gate

1. Distinct foreground/background donors each contribute at least 20%; count adjustment is below
   5%; all counts are exact, renders finite, and grid locking below 1%.
2. Correct composite macro histogram cosine beats wrong-object and pooled-null controls.
3. At least two of three composites remain recognizable and prompt-consistent. Wrong-object must
   visibly alter the class-defining subject in at least two rows.
4. Every target remains source-disjoint and contributes no Jewel rows, tube path, radius, or class
   statistics.

## Decision rule

- **Pass:** proceed to the smallest prompt-to-program learner over scene, persistent track, and
  local phrase decisions.
- **Fail:** the qualitative composition result remains useful, but hand-designed tubes are not a
  sufficient tokenization; learn object slots/masks with direct rendered supervision before a text
  prior.

As before, this is target-program-selected and template-backed mechanism evidence, not prompt-only
generation.

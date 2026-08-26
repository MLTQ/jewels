# Gate 2a9 protocol: compositional trajectory-tube oracle

Frozen after Gate 2a8 showed that one coherent source program restores recognizable ballerina,
dog, and welder structure in all three source-disjoint rows.

## Decision question

Can a persistent token own a connected spacetime region strongly enough to preserve a recognizable
subject while a separate token owns the surrounding scene, yielding a Jewel field that is not any
single retrieved training program?

## Frozen experiment

- Reuse the Gate 2a8 18/9 split, fine 16x16x8 block descriptors, and active K=1,024 Jewel
  vocabulary.
- For a source-disjoint target block program, rank the six eligible same-prompt training fields by
  whole-program distance. Use the nearest as foreground donor and the next-nearest as background
  donor. Neither donor may be the held-out field.
- Derive one connected moving tube from **training-donor disagreement only**. At each of eight time
  slabs, weight foreground/background addressed descriptor distance by a centered Gaussian prior,
  compute its spatial centroid, smooth the eight centroids with a fixed `[1,2,1]` temporal kernel,
  and use a fixed normalized XY radius of 0.78.
- Emit foreground-donor Jewels whose continuous centroids lie inside the moving tube and
  background-donor Jewels outside it. Jitter centroids by 0.005 and adjust uniformly to exactly
  72,000 rows. The target contributes only the donor-selection program, never Jewel rows or the
  tube path.
- Controls: a coherent nearest-source ceiling; a wrong-object arm whose tube comes from the next
  semantic scene while the background remains correct-scene; and a pooled-null coherent source.
  All arms use matched random seeds.

## Gate

1. Both foreground and background donors contribute at least 20% of the pre-adjustment composite;
   they are distinct, and the emitted pair is not any complete training field.
2. Count adjustment is below 15%; every output has exactly 72,000 Jewels, finite renders, continuous
   centroids, and less than 1% exact grid locking.
3. Source-disjoint correct composites beat wrong-object and pooled-null controls in macro
   cell-conditioned active-token histogram cosine.
4. At least two of three correct composites show a recognizable prompt-consistent subject/action.
   The wrong-object arm must visibly alter the central subject in at least two rows.

## Decision rule

- **Pass:** a compositional, persistent Jewel action grammar exists at the source/trajectory/local
  hierarchy. Train a prompt-and-seed model to predict scene, tube path, and local programs; do not
  train independent Jewel emissions again.
- **Fail:** coherent full-source retrieval is sufficient but this fixed trajectory factorization is
  not. Test learned object masks/slots before claiming that trajectory tokens work.

This oracle is target-program-selected and template-backed. Even a pass is mechanism evidence, not
prompt-only generation.

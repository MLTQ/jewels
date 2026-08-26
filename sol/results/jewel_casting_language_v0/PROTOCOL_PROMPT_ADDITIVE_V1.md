# Additive prompt-to-Jewel language Gate 1c protocol

## Decision question

Do the 33 source fields contain enough repeated style/action evidence for any prompt-conditioned
Jewel speaker, or are the neural Gate-1a/1b failures evidence that prompt control itself is absent?

Gate 1c uses the lowest-capacity compositional language model compatible with the corpus. It cannot
memorize a held-out style/action interaction: source-disjoint counts factor into a global prior, a
style effect, and an action-sentence effect. A pass establishes the prompt pathway and attributes
neural overfit to data/model ratio. A failure blocks the compute pitch.

## Frozen additive model

- Same 33 training fields, nine held-out fields, bundle-1 K=1,024 active language, 8,192/16,384
  deterministic Jewel samples, and correct/cyclic-shuffled/null controls as Gate 1b.
- Text API remains two strings: visual style and action sentence. Frozen BGE embeddings resolve each
  string to the nearest training factor phrase; no class/source IDs are accepted.
- Accumulate cell counts and cell-conditional covariance/surface/gradient token counts separately
  for global, style, and action factors.
- Dirichlet posterior mean: token concentration 64 per cell/role and cell concentration 256. Style
  and action posteriors shrink toward the global prior.
- Compose an unseen prompt by product of experts: `log p(style) + log p(action) - log p(global)`,
  normalized separately for cells and every cell/role token distribution.
- Free run samples 72,000 cells, uniform continuous positions inside those cells, and all three marks
  from the composed distributions. The positions are irregular by construction, never cell centers.

## Registered Gate 1c

All checks must pass:

1. Correct teacher-forced cell NLL is below shuffled and null controls.
2. Correct token NLL is below both controls for all three roles; macro improvement over the better
   control is at least 2%.
3. Correct free-run target-histogram cosine exceeds shuffled/null by at least 0.02.
4. Correct generations retrieve their matching held-out target at top 1 for at least two of three
   prompts.
5. Frozen BGE factor resolution returns the declared held-out style and exact action sentence for
   all prompts.
6. Generated centers have below 1% cell-center locking; renders are finite; inference uses only the
   two text strings and declared seed.

PSNR remains diagnostic for independent stochastic samples. This gate proves prompt control of the
native language, not production visual quality or novel-vocabulary text generalization.

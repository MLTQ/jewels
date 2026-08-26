# Shared-scene prompt Gate 1g protocol

## Decision question

Does one shared stochastic scene state fix the source-disjoint prompt speaker's independent-Jewel
averaging failure on the same balanced six-video corpus used by Gate 1f?

This protocol is frozen after Gate 1f failed and before Gate 1g is trained.

## Frozen representation and data

Use the Gate 0f continuous-centroid plus covariance/surface/gradient K=1,024 Jewel language. Train
only the six exact-prompt LTX videos (two independent sources per prompt). Validate on the original
three source videos with three independent field fits each. All Gate 1f source exclusions remain.

## Frozen causal change

Keep the factorized text/coordinate speaker and its optimizer, batch, prompt dropout, early stopping,
72,000-Jewel generation, random seeds, and controls. Add only:

1. a 32-dimensional diagonal Gaussian scene prior predicted from style/action text;
2. a learned diagonal posterior for each training source;
3. KL weight 0.05 from source posterior to text prior;
4. probability 0.25 of decoding a text-prior scene rather than a source-posterior scene per source
   on each training step;
5. one scene draw shared by every centroid and token in a generated program.

Training source identities may index variational posteriors, exactly as an encoder may read training
videos. Inference may receive only style text, action text, and a declared random seed.

## Frozen gate and interpretation

Require the existing Gate 1e likelihood, free-run histogram, retrieval, irregularity, finite-render,
and leakage checks, plus confirmation of one shared scene per program. A pass proves that a bounded
prompt vocabulary can generate source-disjoint native Jewel programs when the model has global
sample state. It does not prove arbitrary-prompt composition, high visual quality, or continuation.

A failure is still decisive if correct-versus-shuffled improves but null remains better: it means
two videos per prompt cannot identify a useful scene prior, and the next experiment must increase
independent videos per exact prompt before changing architecture again.

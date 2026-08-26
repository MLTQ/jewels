# Factorized prompt-native Jewel caster Gate 1b protocol

## Registered response to Gate 1a

Gate 1a produced a small correct-versus-shuffled token signal but the null prior won. Two frozen
causes are addressed:

1. one embedding per full prompt did not expose that styles and action sentences repeat across the
   33-example compositional design;
2. a 64-component global Gaussian mixture could not represent prompt-shaped continuous intensity.

Gate 1b keeps the field split, BGE encoder, active Jewel language, sampling budgets, optimizer,
controls, and prompt-only inference boundary. It changes the speaker architecture before results.

## Frozen factorized speaker

- Style text: `"{style} visual style"`; action text: exact `source_prompt` sentence. Both are encoded
  separately by frozen `BAAI/bge-small-en-v1.5`; no categorical IDs enter the model.
- Separate linear projections of style text and action text are added to a four-frequency continuous
  coordinate representation before a shared four-layer 512-wide trunk.
- The trunk emits three K=1,024 token distributions and a scalar spatial-intensity logit.
- Density training is noise-contrastive: target centroids are positive, matched uniform continuous
  points are negative. Free-run centroids importance-sample 72,000 of 288,000 uniform continuous
  proposals without replacement.
- Null dropout replaces both text vectors with zeros for 10% of training rows.
- Schedule remains AdamW 3e-4, batch 4,096, at most 15,000 updates, evaluation every 500, and ten
  stale evaluations at 0.1% relative improvement.

## Frozen controls and Gate 1b

Correct, cyclically shuffled, and zero-text null arms retain identical validation samples and
declared free-run seeds. All checks must pass:

1. Correct-prompt density NCE is below shuffled and null controls.
2. Correct-prompt token NLL is below both controls for every active role; macro improvement over the
   better control is at least 2%.
3. Correct-prompt free-run active-token histogram similarity to its target exceeds shuffled/null by
   at least 0.02 macro cosine.
4. Correct generated fields retrieve their matching target histogram at top 1 for at least two of
   three held-out prompts.
5. Generated centers have less than 1% exact cell-center locking; every rendered value is finite.
6. Inference inputs are only style text, action text, and declared random seed—no target field,
   centroids, tokens, pixels, latents, class IDs, or source identity.

Random-volume PSNR is retained as a diagnostic but is not gated: independent stochastic video
samples are not expected to align pixelwise with one target realization. Gate 0f already owns
renderability; Gate 1b owns causal text control of the native action language.

# Balanced exact-prompt Gate 1f protocol

## Decision question

Did Gate 1e fail because 57 one-example prompt fields overwhelmed six exact-prompt fields, or
because an independently sampled Jewel process cannot preserve sample-level scene coherence?

This protocol is frozen after Gate 1e r2 failed and before the balanced result is inspected.

## Frozen data

Training is restricted to the six source-disjoint LTX videos registered by Gate 1e: two videos for
each exact held-out prompt. The 57 one-example compositional fields are excluded. Validation remains
the three original sources, each represented by fitter seeds 0, 1, and 2:

- `anime__05_ballet_train_00_seed60500`
- `cartoon__01_dogpark_train_00_seed61100`
- `render3d__04_workshop_train_00_seed62400`

No validation pixels, fields, centroids, Jewel tokens, source IDs, or video latents enter training.

## Frozen model and controls

Keep the Gate 1e factorized speaker, bundle-1 K=1,024 codebook, frozen BGE embeddings, optimizer,
prompt dropout, validation checkpoint selection, 72,000-Jewel free runs, generation seed, and
correct/cyclic-shuffled/null controls unchanged. Only the explicit training-source allowlist differs.

## Interpretation

- The existing Gate 1e checks remain the absolute gate.
- A pass proves bounded-vocabulary, source-disjoint prompt binding for a balanced three-prompt
  corpus. It does not prove novel prompt composition, diversity, long-window continuation, or
  production visual quality.
- If correct text beats shuffled text but not the prompt-blind arm and qualitative samples remain
  texture-like, imbalance is not the primary failure. The next architecture must emit or sample a
  shared scene-level state before emitting individual Jewels.

# `audit_learned_trajectory_speaker.py`

## Purpose

Runs the rendered Gate 2b1 battery using programs sampled from the learned autoregressive speaker
under unseen correct, cyclic-shuffled, and null text.

## Components

### `main`

- **Does**: Restores the best learned speaker, regenerates the training-only semantic realizer,
  samples three seeds for each unseen paraphrase, and renders the matched control suite.
- **Does**: Reuses Gate 2b0's OpenCLIP video embedding and semantic aggregation, while additionally
  checking that unmasked source-token samples belong to the predicted scene.
- **Rationale**: This separates success of the finite grammar from success of learning to speak it.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2b1 protocol | Best-NLL checkpoint and seeds 20260918–20 | Model/evaluation ownership |
| Cross-gate comparison | Renderer, three frames, CLIP model, and semantic thresholds match 2b0 | Evidence parity |
| Claim scope | Learned model is small and source-token-backed, not open vocabulary | Model description |

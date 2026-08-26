# `train_scene_latent_prompt_jewel_caster.py`

## Purpose

Trains Gate 1g, the causal shared-scene repair to the independent Jewel speaker, and evaluates it on
source-disjoint exact prompts with correct, shuffled, and prompt-blind controls.

## Components

### `gaussian_kl`

- **Does**: Aligns a learned per-training-source posterior with the text-conditioned diagonal scene
  prior. Source identity is a training-only variational parameter and is absent at inference.

### `scene_control_metrics`

- **Does**: Tests correct, cyclic-shuffled, and null text at each arm's scene-prior mean on identical
  held-out continuous centroids and active Jewel targets.

### Training and generation

- **Does**: Samples one posterior or prior scene state per source per step, shares it across every
  Jewel from that source, applies text dropout, and checkpoints on held-out correct-prompt loss.
- **Does**: At inference samples one scene from text and a declared seed, then casts 72,000 irregular
  centroids with covariance/surface/gradient tokens through the frozen bundle-1 codebook.
- **Does**: Records that source latents exist only during training and explicitly audits inference
  for target/source leakage.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 1g protocol | Scene dim 32, KL 0.05, prior-use probability 0.25 | Frozen ablation |
| Qualitative renderer | Same generated-program center/token keys as earlier prompt speakers | Artifact schema |
| Scientific review | Correct/shuffled/null share generation noise and differ only by text | Control ownership |

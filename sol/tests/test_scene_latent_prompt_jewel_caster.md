# `test_scene_latent_prompt_jewel_caster.py`

## Purpose

Protects the shared stochastic scene contract before the model is used in a prompt-binding gate.

## Components

### `SceneLatentPromptJewelCasterTests`

- **Does**: Checks scene-prior, density, and three-token output shapes.
- **Does**: Exercises one scene state across all centers and marks in a sampled Jewel utterance.
- **Does**: Rejects prompt/scene/coordinate row mismatches.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Scene-latent Gate 1g | One shared scene per generated program | Sampling semantics |

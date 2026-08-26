# `test_train_scene_latent_prompt_jewel_caster.py`

## Purpose

Protects the variational scene alignment and correct/shuffled/null evaluation used by Gate 1g.

## Components

### `SceneLatentPromptTrainingTests`

- **Does**: Verifies zero KL for identical diagonal Gaussians and rejects mismatched shapes.
- **Does**: Verifies all prompt controls expose density plus covariance/surface/gradient metrics.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 1g | KL and prompt controls retain their mathematical meaning | Loss/control definitions |

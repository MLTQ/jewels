# `scene_latent_prompt_jewel_caster.py`

## Purpose

Defines the smallest causal repair to the independent prompt speaker: sample one continuous scene
state from text and declared randomness, then condition every continuous centroid and active Jewel
token in that utterance on the same state.

## Components

### `SceneLatentPromptJewelCaster`

- **Does**: Predicts a diagonal Gaussian scene prior from separate frozen style/action embeddings.
- **Does**: Adds the shared scene state to continuous-coordinate features before density and
  covariance/surface/gradient token heads.
- **Does**: Samples continuous centroids and three K-way Jewel marks without a target field, source
  pixels, a grid-center lookup, or a source identifier.

## Rationale

Independent per-Jewel noise cannot decide on one subject, layout, or motion for the whole sample;
averaging is then the maximum-likelihood answer. A shared stochastic state is the minimal analogue
of the first scene/window token in an autoregressive Jewel language. It does not make the Jewels a
codec: the public output remains continuous centroids plus predefined Jewel vocabulary entries.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Scene-latent trainer | Prior parameters, shared-state token/density calls, sampling helpers | Method signatures |
| Native Jewel renderer | Output still decodes through the frozen bundle-1 factor codebook | Token semantics |
| Prompt-only audit | Scene comes only from text and declared randomness | Inference inputs |

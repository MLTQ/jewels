# `prompt_jewel_caster.py`

## Purpose

Defines the first native text-to-Jewel speaker. A frozen text vector predicts a continuous mixture
of irregular centroids, then text plus each sampled centroid predicts covariance, surface/opacity,
and gradient vocabulary marks. Decoding goes directly to renderable Jewels—there is no source video
or dense video intermediate.

## Components

### `encode_active_jewel_tokens` / `active_tokens_to_features`
- **Does**: Assigns and decodes the passing three-role individual-Jewel language.

### `active_cell_histogram`
- **Does**: Builds the cell-conditional active-token signature used for prompt-retrieval controls.

### `PromptCentroidGMM`
- **Does**: Predicts and samples a 64-component continuous diagonal centroid mixture from text.
- **Rationale**: This is a genuine irregular point process, not a hidden output lattice.

### `PromptJewelCaster`
- **Does**: Uses text and continuous Fourier coordinates to predict three token distributions;
  joint loss trains token semantics and centroid density, while free sampling declares temperature,
  top-k, and random generator.

### `FactorizedPromptJewelCaster`
- **Does**: Projects frozen style text, action text, and continuous Fourier coordinates separately;
  their additive composition drives both token logits and a nonlinear spatial intensity field.
- **Rationale**: Repeated style/action factors can generalize to an unseen combination, while
  importance sampling from a learned continuous intensity represents shapes that a global Gaussian
  mixture cannot.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 1 trainer | Frozen 384D BGE text vectors and three active K1024 roles | Conditioning/language schema |
| Free-run audit | Only one text vector and random generator enter `sample` paths | Inference boundary |
| Renderer | Decoded features use codebook normalizer and exact sampled centers | Feature contract |
| Gate 1b | Separate style/action text vectors and uniform continuous proposals | Factorized prompt contract |

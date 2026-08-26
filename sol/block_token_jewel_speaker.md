# `block_token_jewel_speaker.py`

## Purpose

Expands one shared discrete token per local spacetime block into continuous centroid density and the
three active Jewel-role tokens. It is the Gate 2 replacement for conditionally independent Jewels
driven only by one global scene vector.

## Components

### `BlockTokenJewelSpeaker`

- **Does**: Combines a block-token embedding, fixed block-address Fourier features, and continuous
  within-block Fourier features.
- **Does**: Predicts local density and covariance/surface/gradient token distributions.
- **Does**: Samples centroids from continuous proposals, then samples the active Jewel roles.
- **Rationale**: Thousands of Jewels in a region share an explicit structural decision without
  exposing a raster grid in the emitted representation.

### `program_tokens`

- **Does**: Routes each continuous centroid to its block's discrete token.
- **Rationale**: Routing is lookup-only; coordinates remain the original irregular samples.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a trainer | One token per 8x8x4 block and separate positive/negative density conditions | Program shape or loss signature |
| Gate 0f decoder | Output is continuous centers plus three K=1024 active tokens | Output schema |
| Future prompt speaker | A serialized block program can drive generation without target rows | Program semantics |
| Irregularity audit | Sampling proposals are continuous and never cell centers | Proposal or routing changes |

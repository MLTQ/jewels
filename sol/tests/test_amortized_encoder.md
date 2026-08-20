# `test_amortized_encoder.py`

## Purpose

Protects the video-to-jewel encoder's parameter, latent, renderer, and gradient contracts.

## Components

### `AmortizedEncoderTests`
- Verifies the precision-Cholesky all-center renderer agrees with the canonical renderer.
- Verifies the support-complete tiled renderer agrees with an explicit five-sigma all-center
  oracle in both output and every differentiable parameter gradient.
- Verifies the video-seeded initialization preserves coarse spatial appearance and that canonical
  feature export has the expected shape.

### `EncodeDecodeSplitTests`
- Verifies generation-facing latents have stable shapes and decode without access to a video.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Encoder convergence runs | Sparse support rendering is numerically and gradient equivalent to the declared finite-support oracle | Support selection or render math |
| Stage-1 latent generator | Encode/decode split remains exact and video-free after encode | Latent schema |

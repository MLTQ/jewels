# `test_latent_prior.py`

## Purpose

Protects the text-conditioned generation seam: the raster prior must train under flow matching and
must satisfy the same nullable-condition velocity interface used by local inpainting.

## Components

### `LatentPriorTests`
- **Does**: Exercises flow-loss gradients, explicit fixed paths, sampling, and conditional/
  unconditional velocity shapes, masked dirty-only paths, plus distribution energy-distance
  behavior.
- **Interacts with**: `latent_prior.py` and indirectly the `inpaint.py` callable contract.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Future text-to-video and editor training | One prior serves full generation and masked repair | Forward signature |

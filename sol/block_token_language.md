# `block_token_language.py`

## Purpose

Defines the discrete local spacetime layer between a scene/window utterance and the proven
continuous-centroid Jewel phrases. Blocks are routing and context only; they never quantize emitted
centroids.

## Components

### `BlockTokenCodebook`

- **Does**: Stores training-owned descriptor normalization and the discrete block prototypes.
- **Does**: Serializes without depending on Python object pickling.

### `block_local_coordinates` / `block_centers`

- **Does**: Maps continuous centroids to block IDs and continuous within-block coordinates.
- **Rationale**: The block address supplies hierarchy while preserving irregular placement.

### `block_serialization_order`

- **Does**: Orders blocks time-major with spatial Morton/Z order inside each time slab.
- **Rationale**: Nearby regions stay nearby in the future autoregressive token stream.

### `block_descriptors`

- **Does**: Summarizes occupancy, fine local density, centroid moments, and normalized intrinsic
  Jewel moments into the preregistered 77-dimensional oracle descriptor.
- **Rationale**: The oracle asks whether compact local shared state can carry structure, not whether
  the text prior can already infer it.

### `fit_block_token_codebook` / `encode_block_tokens`

- **Does**: Fits deterministic Lloyd prototypes on training blocks and assigns one token per block.
- **Rationale**: A finite reusable token is a stricter test than an unconstrained per-source vector.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a trainer | 8x8x4 programs, 77D descriptors, deterministic assignments | Descriptor or ordering changes |
| Block Jewel speaker | Program index matches `GridSpec.cell_index` | Cell layout changes |
| Future transformer | `block_serialization_order` is time-major Morton | Serialization changes |
| Irregularity audit | Within-block coordinates remain continuous | Snapping or quantization |

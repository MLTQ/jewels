# `jewel_casting_language.py`

## Purpose

Defines the first falsifiable Jewel-native language: a cast addresses a coarse spacetime cell,
places a learned local constellation at a continuous anchor, and supplies continuous residuals.
The representation is evaluated as a generative action vocabulary, never as a file codec.

## Components

### `CastingNormalizer` / `CastingBundles`
- **Does**: Fits train-owned intrinsic moments and canonically groups every Jewel into lossless
  cell-local bundles.
- **Rationale**: Absolute location remains continuous; motifs learn relative geometry and
  appearance without silently dropping difficult targets.

### `fit_motif_codebook` / `MotifCodebook`
- **Does**: Fits deterministic Lloyd prototypes over bundle vectors and reports utilization,
  perplexity, and assignment error.
- **Does**: Retains the fitted count coordinate for assignment while program syntax remains the
  authoritative decoded count.
- **Rationale**: A reusable casting vocabulary must explain recurring local constellations rather
  than memorize absolute centroids.

### `encode_program` / `decode_program`
- **Does**: Converts a continuous field to motif casts plus residuals and casts it back at a chosen
  residual scale.
- **Rationale**: Sweeping residual scale exposes how much visual work the token actually owns.

### `program_histogram` / `histogram_cosine`
- **Does**: Compares cell-addressed motif distributions without requiring unstable one-to-one Jewel
  correspondence.
- **Rationale**: Equivalent decompositions may differ at the primitive level while sharing a
  learnable local language.

### `quantize_centers_to_cells`
- **Does**: Supplies the explicit grid-locking negative control.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate-0 audit | Canonical feature layout is 22 values and no Jewel is dropped | Feature or count semantics |
| Future caster | Program contains cell, motif, count, anchor, and residual state | Program schema |
| Irregularity gate | Cell address never replaces the continuous anchor | Center decode |

## Notes

- The current Lloyd fit is evidence code. A trainable EMA/VQ vocabulary is downstream of Gate 0.
- Padded rows exist only inside partial bundles; `counts` prevent them from becoming Jewels.

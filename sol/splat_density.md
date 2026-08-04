# `splat_density.py`

## Purpose

Measures how many spacetime jewels can meaningfully affect each video frame. It replaces raw total
jewel count and broad sigma-support intersections with contribution-aware density evidence.

## Components

### `FrameSplatDensity`

- **Does**: retains per-frame support, peak-alpha threshold, and participation-ratio counts
- **Rationale**: raw vectors allow transparent aggregation across examples without averaging
  already-averaged summaries

### `measure_frame_splat_density`

- **Does**: reconstructs marginal temporal variance from gauge-free log covariance and measures
  per-frame 3σ support, potential peak alpha above declared thresholds, and alpha participation
- **Interacts with**: the 22-D feature contract in `stprim/prior/featurize.py`
- **Rationale**: a splat whose temporal support touches a frame may still be too faint to supply
  useful spatial detail; peak alpha is its best possible coefficient anywhere in that frame

### `temporal_standard_deviation`

- **Does**: recovers each jewel's marginal temporal sigma from canonical log covariance
- **Interacts with**: `streaming.py`, which turns finite temporal support into stable lifecycles
- **Rationale**: density and streaming ownership must use the same lifespan geometry

### `summarize_counts`

- **Does**: emits JSON-safe mean, median, minimum, and maximum counts

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `audit_splat_density.py` | Per-frame tensors on the input device | Field names or threshold keys |
| `streaming.py` | Public temporal sigma uses canonical covariance semantics | Function name or units |
| Density experiments | Feature layout is `mu(3), logSigma(6), ..., logit_w(1)` | Feature schema |

## Notes

- Peak-alpha counts are an upper bound on visible spatial contributors: they evaluate each splat at
  its best spatial point on the frame plane, not at every pixel.
- Participation ratio measures temporal/opacity concentration, not independent image detail.

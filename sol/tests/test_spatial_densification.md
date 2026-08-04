# `test_spatial_densification.py`

## Purpose

Protects the temporal-preserving split policy intended to turn total jewel growth into real
per-frame detail.

## Components

### `SpatialDensificationTests`

- **Does**: verifies that an identity-oriented split keeps temporal center/scale, shrinks both
  spatial axes by √2, conserves combined Gaussian volume, and rejects unknown policies
- **Interacts with**: `stprim/fit/adapt.py` and `PrimitiveField`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| 90k spatial control | Two children do not lose temporal lifespan | Split geometry |
| Recovery methodology | Split mode is explicit rather than inferred | Policy validation |

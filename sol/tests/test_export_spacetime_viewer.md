# `test_export_spacetime_viewer.py`

## Purpose

Protects the browser export schema and the covariance-to-slice derivation using a small synthetic
checkpoint with analytically known axis-aligned geometry.

## Components

### `test_payload_has_stable_schema_and_strides`

- **Does**: Verifies field metadata, provenance, and flat-array lengths.

### `test_axis_aligned_slice_derivatives_match_geometry`

- **Does**: Verifies zero conditional motion, 2D slice covariance, temporal sigma, and quaternion
  normalization/order.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `export_spacetime_viewer.py` | Stable schema and mathematically correct slice arrays | Schema or covariance changes |
| Three.js viewer | Quaternions are `x,y,z,w` and arrays retain declared strides | Reordering fields |

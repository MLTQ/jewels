# `test_streaming_features.py`

## Purpose

Protects the affine coordinate contract used by continuation training and rendering.

## Components

### `StreamingFeatureTests`

- **Does**: verifies feature round-trip accuracy and render invariance with tilted covariance and P1
  color gradients
- **Interacts with**: `streaming_features.py` and the exact reference renderer

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Continuation evaluation | Local predictions restore the same global field | Tensor transform rules |

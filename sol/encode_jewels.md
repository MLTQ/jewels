# `encode_jewels.py`

## Purpose

The inference half of the dense-intermediate pivot: one forward pass turns a video window into
a canonical jewel field on disk, in the exact `_generated_field.pt` schema every existing audit
tool consumes (`perceptual_eval.py --field`, `render_exact`, the edit machinery).

## Components

### `main`
- **Does**: Validates checkpoint architecture and manifest ownership, encodes each window of
  the requested split at fit geometry, converts predictions to canonical features (no-grad
  eigh), and writes per-source field files plus an encode summary with checkpoint provenance.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| jewels-sb0 gate | Field files named `<source_id>_generated_field.pt` with features/background | Schema |
| `perceptual_eval.py` | Canonical 22-D features render under `render_exact` unchanged | Layout |

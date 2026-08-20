# `test_build_encoder_manifest.py`

## Purpose

Protects deterministic corpus construction and the fixed validation split used by encoder scaling
experiments.

## Components

### `BuildEncoderManifestTests`
- Verifies only completed clips enter the manifest, evaluation prompts are held out, and style
  prefixes keep source IDs unique.

### `SubsampleTrainTests`
- Verifies nested prefixes retain validation rows.
- Verifies the 12-example prefix covers every semantic class and rotates across all five styles.
- Verifies the 60-example prefix covers every style/class pairing exactly once before repeats.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Encoder convergence curve | Training subsets are deterministic, nested, and diversity-balanced | Prefix ordering |
| Validation comparisons | Held-out rows are identical at every training size | Split filtering |

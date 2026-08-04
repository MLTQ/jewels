# `corpus.py`

## Purpose

Connects one or more production fitted-checkpoint corpora to the research tokenizer while enforcing
source-video-level holdout splits. It prevents adjacent windows from one video appearing in both
training and validation and preserves a domain label for controlled transfer experiments.

## Components

### `load_fitted_corpus`

- **Does**: Loads one or more roots through the production gauge-free featurizer and records source,
  domain, background, and per-window shape metadata.
- **Rationale**: Tokenization uses normalized `(u, v, t)` coordinates, so differently shaped videos
  can share a model as long as their jewel counts and feature schema agree.
- **Interacts with**: `stprim/prior/featurize.py` without changing its CLI-oriented import layout.

### `split_by_source`

- **Does**: Deterministically holds out complete source videos, either by seeded count or exact IDs.
- **Rationale**: Random window splits would leak nearly adjacent footage.

### `FeatureNormalizer`

- **Does**: Fits train-only statistics and standardizes feature dimensions 3 onward; optionally
  averages per-domain moments so a one-window domain is not numerically erased by a large corpus.
- **Rationale**: Center coordinates remain physical `[-1,1]` so raster bucketing and cell-constrained
  decoding keep exact spatial meaning.

### `FittedExample` / `SourceSplit`

- **Does**: Carry the minimal typed data contracts for training and validation.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `train_dense_autoencoder.py` | Uniform count, per-example shape, domain ID, train-only normalization | Data fields |
| `evaluation.py` | Raw canonical features and video shape | Feature semantics |
| Checkpoints | Normalizer state is saved with model weights | Normalization policy |

# `test_evaluation.py`

## Purpose

Protects held-out render evaluation against early-training edge cases such as poor covariance and
count predictions. Even an untrained model must produce a finite, structured report.

## Components

### `EvaluationTests`
- **Does**: Runs a tiny CPU encode/decode/render evaluation and checks report validity.
- **Interacts with**: `evaluation.py`, `autoencoder.py`, `corpus.py`, and `render.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Training CLI | Evaluation cannot crash before the model learns sensible jewels | Report behavior |

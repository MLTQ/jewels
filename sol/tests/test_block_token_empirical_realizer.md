# `test_block_token_empirical_realizer.py`

## Purpose

Protects the finite macro-token realization contract used by Gate 2a2.

## Coverage

- Training phrases are pooled without loss into token-indexed reservoirs.
- Sampling emits exactly the requested number of continuous centers and correlated active tokens.
- Empirical role likelihood covers covariance, surface, and gradient.
- The prompt-blind control selects a token with realizable mass.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a2 audit | Exact count, finite NLL, continuous coordinates | Fit or sample semantics |

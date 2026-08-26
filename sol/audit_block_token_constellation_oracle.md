# `audit_block_token_constellation_oracle.py`

## Purpose

Runs Gate 2a4: determine whether a K=1024 block token that casts one complete predefined medoid
constellation preserves the joint geometry lost by independent empirical tuple sampling.

## Components

### `adjustment_macro`

- **Does**: Reports how much exact-72k normalization changed assembled constellation counts.
- **Rationale**: A result that depends on extreme resampling would not prove useful token semantics.

### Audit pipeline

- **Does**: Fits medoids only from 18 training fields, assigns held-out oracle programs through the
  frozen vocabulary, and reuses matched oracle/shuffled/null likelihood and render controls.
- **Does**: Applies unchanged global-posterior numerical gates and leaves the preregistered
  constellation-level qualitative decision explicit.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a4 protocol | K=1024 complete medoids, jitter 0.005, exact 72k output | Experiment semantics |
| Gate 2b decision | Numeric gate, count adjustment, and qualitative sheet | Report schema |
| Scientific review | Target selects tokens but contributes no template points | Leakage audit |

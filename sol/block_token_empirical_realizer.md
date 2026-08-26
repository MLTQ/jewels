# `block_token_empirical_realizer.py`

## Purpose

Implements the user's literal macro-token idea: one predefined block token expands into a reusable
empirical collection of continuous local centroid positions and correlated Jewel-role tokens. This
tests finite token semantics without asking an independent neural point sampler to rediscover the
token's geometry.

## Components

### `EmpiricalBlockRealizer`

- **Does**: Stores local Jewel tuples in a compact token-indexed CSR reservoir.
- **Does**: Allocates exactly the requested Jewel count across program blocks using learned mean
  occupancy, samples correlated local position/role tuples, and adds small continuous jitter.
- **Does**: Reports smoothed per-role empirical token NLL for oracle/shuffled/null programs.

### `fit_empirical_block_realizer`

- **Does**: Pools phrases and statistics only from blocks in the registered 18-source training set.
- **Rationale**: A held-out target selects a predefined macro token but never contributes phrases
  to its realization reservoir.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a2 audit | K=256 programs and smoothing 0.1/jitter 0.01 | Reservoir or likelihood semantics |
| Gate 0f decoder | Sample returns continuous centers plus three active tokens | Output schema |
| Irregularity audit | Jittered positions remain within addressed blocks, never at centers | Sampling changes |
| Future Gate 2b | Any predicted block sequence can be realized without a target video | Target leakage at sample time |

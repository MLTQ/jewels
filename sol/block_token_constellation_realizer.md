# `block_token_constellation_realizer.py`

## Purpose

Implements a literal token-to-predefined-Jewels language: each local block token owns one complete
training-medoid constellation of continuous relative centroids and correlated active Jewel roles.
Unlike the empirical reservoir realizer, it preserves the joint spatial arrangement within a block.

## Components

### `ConstellationBlockRealizer`

- **Does**: Casts all tuples in each addressed medoid constellation, adds small continuous jitter,
  and performs one declared global adjustment to exactly 72,000 Jewels.
- **Does**: Reports the unadjusted count and adjustment fraction for every generation.
- **Does**: Supplies smoothed role NLL from each complete medoid template.

### `fit_constellation_block_realizer`

- **Does**: Chooses the training block occurrence nearest each frozen K=1024 descriptor prototype.
- **Does**: Stores the chosen complete point sets in token-indexed CSR form.
- **Rationale**: A medoid is a real coherent constellation; averaging or independently mixing rows
  would erase the dependency this gate is designed to test.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a4 audit | One minimum-distance medoid per utilized K=1024 token | Medoid rule |
| Gate 0f decoder | Samples return continuous centers and three active Jewel tokens | Output schema |
| Scientific review | Only 18 training fields contribute templates | Fit data ownership |
| Irregularity audit | Jitter 0.005 and open-boundary clamping | Coordinate semantics |

# `audit_individual_jewel_language.py`

## Purpose

Tests the original proposal directly on fresh fields: one Jewel is described by covariance,
surface/opacity, and gradient vocabulary tokens plus its exact continuous centroid. The constant
bundle-1 layout coordinate is neither emitted nor allowed to dominate similarity.

## Components

### `active_individual_histogram` / `pairwise_active_similarity`
- **Does**: Concatenates cell-conditional histograms for the three nonconstant roles and compares
  independent fits of the same video with unrelated videos.

### `main`
- **Does**: Exhaustively encodes nine fresh fields with the frozen source-disjoint codebook, renders
  token-only and numerical full-residual arms, audits decision count/tilt/irregularity, and evaluates
  the frozen Gate-0f checks.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 0f protocol | Active roles covariance/surface/gradient; layout is constant and absent | Language schema |
| Future caster | Three token IDs plus cell and continuous centroid per Jewel | Emission contract |
| Scientific review | Exact residual is audit-only and never part of token render | Candidate ownership |

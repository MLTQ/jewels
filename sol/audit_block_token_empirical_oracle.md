# `audit_block_token_empirical_oracle.py`

## Purpose

Runs Gate 2a2/2a3 against the frozen K=256 or preregistered K=1024 block language. It tests whether a block token that directly
maps to predefined local Jewel/centroid tuples preserves structure better than the failed
conditionally independent neural point sampler.

## Components

### `empirical_control_metrics`

- **Does**: Computes smoothed covariance/surface/gradient NLL for oracle, shuffled, and null block
  programs on direct and source-disjoint fields.

### `evaluate_generation`

- **Does**: Realizes exactly 72,000 continuous Jewels under matched seeds, computes language/render
  audits, and writes direct and source-disjoint time-progress sheets.
- **Does**: Preserves optional realization diagnostics such as constellation count adjustment.

### `realizer_device`

- **Does**: Resolves the tensor device for either empirical-reservoir or complete-constellation
  realizers without changing their public representation.

### Gate assembly

- **Does**: Loads the immutable K=256/K=1024 block checkpoint and global-posterior baseline, fits empirical
  reservoirs only from the 18 registered sources, and applies unchanged Gate 2a thresholds.
- **Does**: Marks target block assignment as oracle leakage and requires a separate qualitative
  locality review before advancement.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a2/2a3 protocol | K in {256,1024}, smoothing 0.1, jitter 0.01, unchanged thresholds | Experiment semantics |
| Gate 2b decision | Numeric gate plus qualitative locality evidence | Report gate schema |
| Scientific review | Sampling uses no target rows; only oracle program selection leaks | Leakage audit |

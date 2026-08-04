# `domain_sampling.py`

## Purpose

Select training examples from mixed fitted corpora without allowing a large fixed-camera domain to
overwhelm a small transfer domain.

## Components

### `sample_domain_balanced_indices`

- **Does**: cycles through sorted domain IDs and samples a random example within each chosen domain
- **Rationale**: with batch one and two domains, steps alternate exactly 50/50; stochastic weighted
  sampling could still starve the one-window domain over short diagnostics
- **Interacts with**: `train_dense_autoencoder.py`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Dense tokenizer trainer | Returned CPU indices address the prepared training list | Ordering or return dtype |
| Reproducibility tests | Domain sequence is round-robin; within-domain choice uses caller RNG | Step convention or RNG use |

# `train_block_token_oracle.py`

## Purpose

Runs Gate 2a: fit a finite local block vocabulary, train its shared Jewel expander, and determine
whether target-derived local state solves the structural bottleneck left by a global scene vector.
The oracle is deliberately leaky and is never reported as prompt-only generation.

## Components

### `BlockOracleBatch` / `make_batch`

- **Does**: Samples deterministic source-owned continuous centroids, negative coordinates, and
  frozen Gate 0f Jewel targets without dropping fields.

### `resolve_split_arguments`

- **Does**: Reuses the immutable Gate 1h report's exact roots and source split, or validates a fully
  explicit split.
- **Rationale**: A single report-owned split prevents command-line transcription drift.

### `cyclic_shuffled_programs`

- **Does**: Aligns each field with the same-rank fit from the next prompt source.
- **Rationale**: The shuffled control changes local structural state while preserving token budget.

### `oracle_control_metrics`

- **Does**: Measures active-token NLL and density NCE for oracle, shuffled, and null block programs.

### `evaluate_generation`

- **Does**: Free-runs 72,000 continuous Jewels per arm under matched randomness, audits histogram,
  render, and grid-lock behavior, and writes qualitative time-progress sheets.

### Training and gate

- **Does**: Fits the vocabulary on 18 exact-prompt sources, checkpoints on nine source-disjoint
  fields, compares the three direct sources with the frozen global posterior oracle, and applies
  the preregistered kill/advance checks.
- **Does**: Stores the ordered block programs and time-major Morton serialization for Gate 2b.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a protocol | K in {64,256,1024}, 20k-step ceiling, 12-eval patience, frozen baselines | Schedule or thresholds |
| Evidence aggregator | Stable report sections for teacher forcing, generation, baselines, and gate | JSON schema |
| Future Gate 2b | Portable block codebook, model, program ordering, and null token | Checkpoint schema |
| Scientific review | Target leakage is explicit and promptability is not claimed | Inference audit wording |

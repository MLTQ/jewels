# `train_factorized_prompt_jewel_caster.py`

## Purpose

Trains Gate 1b: an explicitly compositional style-text/action-text Jewel speaker with a continuous
spatial-intensity field. It retains correct/shuffled/null likelihood controls and prompt-only 72k
free runs, while treating independent-sample PSNR as a diagnostic rather than semantic gate.

## Components

### `FactorPromptBatch` / `factor_control_metrics`
- **Does**: Owns fixed positive/negative continuous coordinates and evaluates per-role token NLL plus
  density NCE under correct, cyclic-shuffled, and zero-text controls.

### Factorized training
- **Does**: Encodes unique style strings and action sentences separately through frozen BGE,
  balances sources, applies zero-text dropout, and stops at the registered validation plateau.
- **Does**: Optionally enforces a minimum number of source-disjoint, exact style/action text matches
  per validation prompt before training. The observed counts are recorded in the report, preventing
  semantically related class labels from being mistaken for exact prompt repetition.
- **Does**: Optionally accepts an explicit `--training-source` allowlist so balanced-corpus
  ablations cannot silently include the larger one-example-per-prompt background corpus. Missing
  registered sources fail before training and the frozen allowlist is recorded in the report.

### Prompt-only generation and retrieval
- **Does**: Importance-samples continuous centroids, samples active marks, renders all arms, compares
  target histograms, and tests whether correct generations retrieve their matching held-out prompt.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 1b protocol | Separate style/action text, NCE negatives, continuous proposals | Speaker architecture |
| Scientific review | PSNR remains diagnostic; histogram retrieval owns prompt semantics | Gate definition |
| Pitch artifacts | Factorized caster/program checkpoints and v1 report | Artifact schema |
| Exact-prompt gates | `--minimum-exact-prompt-sources` is checked against checkpoint metadata | Preflight semantics |
| Balanced prompt ablations | `--training-source` is a strict allowlist after validation exclusion | Split ownership |

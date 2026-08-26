# `audit_scene_posterior_oracle.py`

## Purpose

Runs a deliberately leaky causal oracle to distinguish a weak text-to-scene prior from a weak local
independent-Jewel decoder. This is a diagnostic, never a prompt-generation result.

## Components

### `select_oracle_sources`

- **Does**: Selects the first registered training source for each exact `(style, action)` prompt and
  verifies checkpoint posterior rows align with report-owned source order.

### Posterior/prior/null audit

- **Does**: Compares the training-source posterior, correct text-prior mean, and prompt-blind prior
  on the same source-owned centroids/tokens and same free-generation seed.
- **Does**: Renders target, oracle, text-prior, and null programs across time and labels the oracle's
  source leakage explicitly.

## Interpretation

If the posterior oracle improves token NLL by at least 2% and free-run histogram match by at least
0.02 over the text prior, the text scene prior is the next bottleneck. Otherwise even a source-owned
global vector cannot make the conditionally independent Jewel decoder express the scene, so local
hierarchical/block tokens are required.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 1i protocol | Training posterior is diagnostic leakage, never valid inference | Interpretation |
| Hierarchical speaker decision | Frozen 2% NLL and 0.02 histogram oracle margins | Thresholds |

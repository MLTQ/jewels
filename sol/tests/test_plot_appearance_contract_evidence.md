# `test_plot_appearance_contract_evidence.py`

## Purpose

Protects the bounded/residual arm mapping and lower-is-better LPIPS direction used by the appearance-
contract evidence graph.

## Components

### `AppearanceContractEvidenceTests`
- **Does**: Verifies screen-directory names, audit seed mappings, and response-vs-residual-control
  per-style deltas.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Evidence graph | Bounded and residual audits retain their frozen candidate order | Arm mapping |
| Experiment report | Positive LPIPS delta means improvement | Sign convention |

# `test_scaffold_appearance_adapter.py`

## Purpose

Protects the compact RGB adapter's size, zero-start, gating, and strict feature-ownership contracts.

## Components

### `ScaffoldAppearanceAdapterTests`

- **Does**: Confirms the default adapter is far smaller than the full flow, its initialized output is
  zero, the paired base exactly reproduces ordinary sampling, active rows change only RGB, zero-gate
  rows remain wholly identical, and top-fraction gating has deterministic cardinality.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Adapter experiment | Initial adapter is an exact frozen-base control | Head initialization |
| Lifecycle/density gate | Only RGB dimensions 9--11 can differ | Sampler copy policy |
| Saliency screen | Top-20% gate selects a declared number of cells | Rounding policy |
| Compute claim | Adapter remains below 100k parameters | Architecture size |

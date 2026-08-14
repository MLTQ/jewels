# `test_checkpoint_transfer.py`

## Purpose

Protects the distinction between guarded same-manifest initialization and explicit cross-manifest
model transfer.

## Components

### `CheckpointTransferTests`

- **Does**: Creates tiny checkpoint fixtures and verifies exact model-weight loading, serialized
  transfer provenance, fresh-optimizer policy, the same-manifest guard, and rank rejection.
- **Interacts with**: `checkpoint_transfer.py`.
- **Augmentation check**: Confirms same-manifest base weights load exactly while newly declared
  module prefixes retain their constructor state and receive explicit provenance.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Corpus adaptation | Cross-manifest loading requires explicit opt-in | Transfer policy |
| Rank topology | A checkpoint cannot cross per-cell capacity | Compatibility validation |
| Recovery | Model transfer never masquerades as optimizer resume | Provenance schema |
| Architecture spike | Missing state is allowed only below named new-module prefixes | Load policy |

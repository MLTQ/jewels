# `train_hierarchical_jewel_decoder.py`

## Purpose

Trains and source-disjointly audits Gate 0e: a bounded learned continuous head that restores
correlations omitted by independent product tokens. It consumes only the hierarchical phrase and
irregular anchor, never the exact residual or source video.

## Components

### `dataset_loss` / `predict_values`
- **Does**: Evaluates exhaustive active-row correction loss and decodes complete fields in bounded
  chunks. The registered loss baseline is the exact zero-correction product decoder, never a random
  neural initialization.

### Training dataset construction
- **Does**: Samples an equal pair budget from each of 33 training sources, while encoding every pair
  from all nine validation fields.
- **Rationale**: Source balance prevents high-cast or replicated fields from owning the decoder.

### Plateau-controlled training
- **Does**: Trains the frozen architecture, evaluates train and validation every 500 updates, keeps
  the best source-disjoint checkpoint, and stops only after the registered patience condition.

### Render audit and gate
- **Does**: Compares raw product and learned decodes on identical source-owned random volume points,
  audits tilt and center locking, records the model input boundary, and saves the checkpoint/report.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 0e protocol | 33/9 source split, balanced 4096-pair samples, fixed schedule | Experimental ownership |
| Future caster | `decoder.pt` architecture and frozen codebook paths | Checkpoint schema |
| Scientific review | Target values occur only in loss/evaluation code | Forward input boundary |

# `test_corpus.py`

## Purpose

Protects the empirical methodology: windows from one source video never cross the train/validation
boundary, mixed domains retain their identity, and feature normalization cannot distort physical
jewel centers.

## Components

### `CorpusTests`

- **Does**: Tests randomized and exact source holdouts, normalization round-trip behavior,
  equal-domain normalizer weighting, and loading equal-count corpora with different video shapes.
- **Interacts with**: `corpus.py` and `synthetic.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Held-out experiments | Source IDs are disjoint across partitions | Split semantics |
| Token grid | Normalized centers equal raw centers exactly | Normalization policy |
| Joint-domain experiment | Each root has a stable domain ID and may retain its own shape | Loader schema |
| Balanced training | A small domain contributes equally to appearance statistics | Domain weighting |

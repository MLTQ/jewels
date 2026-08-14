# `build_scaling_subsets.py`

## Purpose

Produces the manifests for the realizer data-scaling curve: identical protocol, identical
validation set, and only the number of class-balanced fitted training sources varies. Point N of
the curve trains on every class's source groups `<= N`; the full manifest is the existing
12-source point, so only smaller subsets are ever generated.

## Components

### `build_subset_manifest`
- **Does**: Copies the manifest, keeps every validation example, filters train examples by
  `source_group <= max_source_group`, refuses unbalanced or class-dropping subsets, and records
  a `scaling_subset` provenance block.
- **Rationale**: Group indices are the corpus's own class-balanced ordering, so subset membership
  is deterministic and needs no sampling decisions.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `encode_prompt_manifest.py` | Subset is a complete, valid prompt manifest with a fresh digest | Schema fields |
| Scaling-curve trainers | Validation examples identical across every curve point | Filter rule |
| Results provenance | `scaling_subset` block records group cutoff and counts | Provenance schema |

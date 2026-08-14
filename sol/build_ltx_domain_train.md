# `build_ltx_domain_train.py`

## Purpose

Constructs the domain-matched training contract (roadmap item 18): the twelve LTX train
generations become fitted train rows and the four evaluation generations remain the validation
rows, so the realizer trains and evaluates in the same teacher-generated domain with a
physically disjoint, honestly labelled split — unlike the cel adaptation's same-field aliases.

## Components

### `build_domain_manifest` (`mix_ucf_train` option)
- **Does**: With the flag, each class's original UCF train rows precede its LTX rows verbatim
  (24 sources, 108 views), keeping first-occurrence prompt order so the cache still rebinds
  bit-identically. Used by the divergence diagnosis to test whether corpus diversity restores
  macro-layout.

### `build_domain_manifest`
- **Does**: Walks classes in UCF-manifest order, emits three train rows (prompt_index order) and
  one validation row per class from the LTX corpus manifest, validates each generation's source
  prompt belongs to its class/split prompt set, and stamps `validation_is_unseen=true`,
  `source_overlap=false`, and the LTX corpus's recorded source digest.
- **Rationale**: Class-then-role ordering reproduces the UCF manifest's unique-prompt collection
  order exactly, which is what lets the parent cache's embeddings rebind bit-identically.

### `main`
- **Does**: Verifies the parent cache owns the UCF manifest, writes the manifest with the frozen
  72k/49-frame fit contract recorded, and saves the rebound prompt cache under the new digest.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Realizer/topology trainers | 12 train + 4 validation rows, 49 frames, fits named by stem | Split or naming |
| Prompt provenance | Embeddings identical to the UCF cache rows | Re-encoding |
| Scaling comparison | Same evaluation battery applies unchanged | Protocol drift |

## Notes

- Train views per source drop from six (96-frame UCF fields) to three (49-frame LTX fields), so
  the 12-source domain-matched corpus carries 36 views versus UCF's 72 — report comparisons
  with that asymmetry stated.

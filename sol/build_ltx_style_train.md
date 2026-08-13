# `build_ltx_style_train.py`

## Purpose

Builds the four-field cel/rotoscoped style-adaptation corpus requested for the scaffold generator,
with an explicit same-field reconstruction audit instead of claiming an unseen validation split.

## Components

### `build_ltx_style_manifest`

- **Does**: Replaces the original UCF fields with exactly one completed 49-frame LTX field per
  prompt class. Each physical field receives a unique training alias and validation alias.
- **Rationale**: The existing corpus contract requires every prompt class in both splits, while
  only one cel field currently exists per class. Duplicating ownership aliases retains trainer
  safeguards and makes the unavoidable overlap machine-readable.
- **Validation**: Requires the exact canonical or source-file UCF digest, one completed
  evaluation-role LTX video per validation class, and exact source-prompt agreement. Both UCF
  digests are recorded because the LTX generator receipts hash the serialized source file while
  prompt caches hash canonical JSON.
- **Honesty contract**: `validation_is_unseen=false`, `source_overlap=true`, paired source IDs, and
  `same-field-training-reconstruction` prevent this audit from being reported as generalization.

### `main`

- **Does**: Rebinds the existing frozen prompt vectors to the new manifest digest and atomically
  writes the manifest/cache pair.
- **Interacts with**: `prompt_embeddings.py`, `streaming_corpus.py`, and both scaffold trainers.
- **Rationale**: No text is re-encoded; prompt strings, embedding order, and encoder identity remain
  fixed while field ownership changes explicitly.
- **Single-field mode**: `--class-name` retains only one physical field's paired aliases and slices
  the frozen prompt cache to that class's exact rows for a deliberately narrow memorization gate.

### `select_ltx_style_class`

- **Does**: Validates and extracts one train/reconstruction alias pair sharing a physical field.
- **Rationale**: The capacity/objective gate must not silently train on the other three cel fields.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Field loader | Both aliases point to the same video stem and therefore the same exact 72k fit | Alias policy |
| Mark trainer | Four training and four reconstruction sources each contain initial plus two continuation views | Frame/stride policy |
| Topology trainer | Every class exists in both manifest splits | Split construction |
| Prompt conditioning | Three train prompt variants optimize each class; evaluation prompt audits reconstruction | Prompt ownership |
| Single-field overfit | Exactly one physical stem appears once in each split | Selection policy |

## Notes

- This is a style-adaptation/overfit gate, not evidence of novel prompt or video generalization.
- The next honest generalization gate requires additional cel fields per class or new held-out
  classes with enough training coverage to separate appearance style from action identity.

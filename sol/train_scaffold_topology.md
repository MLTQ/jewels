# `train_scaffold_topology.py`

## Purpose

Trains the first scaffold-conditioned discrete topology head on the 12 UCF fitted fields and
evaluates it without retraining on the four prompt-generated LTX fitted fields. It includes initial
generation and sequential carry, but keeps mark synthesis oracle-controlled.

## Components

### `PreparedTopologyView` / `PreparedTopologySource`

- **Does**: Bind a persistent field stride to aligned RGB guide cells and target carried-state
  channels while retaining whole-source split ownership.

### `_prepare_sources`

- **Does**: Loads each manifest video, builds every complete initial/continuation topology view,
  and aligns its guide/carry rasters.
- **Interacts with**: `streaming_corpus.py`, `scaffold_topology_data.py`, and `video_guide.py`.

### `_mean_counts_by_index` / `_calibrate`

- **Does**: Build a per-stride train-mean baseline and choose the occupancy threshold strictly from
  training fields.
- **Rationale**: Nearly full coarse occupancy makes a strong explicit prior control essential.

### `_diagnostic_split_override`

- **Does**: Optionally holds out explicitly named whole sources and treats every other loaded field
  as training data.
- **Rationale**: The flag supports small, clearly labelled source-level cross-validation studies of
  domain adaptation without rewriting the authoritative manifest or prompt cache.
- **Safety**: This is a diagnostic-only mode. It validates exact source IDs, rejects duplicates and
  all-validation splits, and records `diagnostic_source_cross_validation` in checkpoint metadata.

### `main`

- **Does**: Trains batched count fields, records correct/shuffled/null/no-carry controls, saves a
  resumable checkpoint, and performs three-stride oracle-mark LTX rollouts with exact stable-ID
  carry and density audits.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Leakage-safe gate | Only UCF `train` rows update weights or calibrate threshold | Split policy |
| Diagnostic cross-validation | `--diagnostic-validation-source-id` holds out whole named sources; all other loaded sources train | Diagnostic split policy |
| LTX evaluation | 49 frames yield frontiers 0, 16, and 32 | Stride/window policy |
| Recovery | Checkpoint stores optimizer/scaler/model and exact manifest/grid arguments | Save schema |
| Next realizer coupling | Decoded counts/ranks share the frozen `16×16×8` topology | Grid semantics |

## Notes

- The validation controls use target carry to isolate the current raster decision. The sequential
  rollout instead rebuilds carry only from previously emitted oracle-matched jewels.
- False-positive ranks are recorded but not materialized in oracle rollouts; full mark coupling is
  the next gate if topology recall/selectivity passes.
- Topology checkpoints use 1,024 ranks per cell because initial UCF state peaks at 919. The existing
  continuation realizer was trained with a 512-rank normalization; initial mark coupling therefore
  needs an explicit compatibility experiment rather than silent clipping.
- The default remains the leakage-safe manifest split. Any run using the diagnostic override must
  be reported as cross-validation and must not replace the untouched four-source LTX evaluation.

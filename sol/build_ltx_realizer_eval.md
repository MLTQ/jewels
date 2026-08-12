# `build_ltx_realizer_eval.py`

## Purpose

Builds the cross-domain evaluation contract for applying a UCF-trained video-to-jewel realizer to
the four prompt-generated LTX-2.3 scaffolds without allowing generated validation clips into
normalization or training.

## Components

### `build_ltx_realizer_manifest`

- **Does**: Preserves the 12 original UCF training rows in their exact order and replaces each UCF
  validation row with the completed, same-class LTX evaluation-prompt clip.
- **Rationale**: Keeping the training rows and prompt order unchanged makes the train-only feature
  standardizers and frozen text embeddings identical to the trained checkpoint. Only the held-out
  video/fitted-field distribution changes.
- **Validation**: Requires one completed evaluation scaffold per class and exact agreement between
  its source prompt and the original unseen evaluation template. The LTX corpus's recorded source
  digest must match the exact UCF prompt manifest.

### `main`

- **Does**: Loads both source manifests, rebinds the already encoded unit prompt vectors to the
  derived manifest, and atomically writes the JSON manifest and prompt sidecar.
- **Interacts with**: `prompt_embeddings.py`, `streaming_corpus.py`, and
  `render_prompted_mark_flow.py`.
- **Rationale**: No prompt is re-encoded and no embedding is anonymously copied; the derived cache
  receives a new manifest digest and explicit per-example ownership.
- **Provenance check**: The input prompt cache must itself match the source UCF manifest before its
  vectors can be rebound.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Realizer evaluation | 12 original UCF train rows and four LTX validation rows | Split construction |
| Train normalization | Training row order/content is unchanged | Source replacement policy |
| Prompt provenance | Prompt strings/order and encoder metadata remain identical | Cache rebinding |
| LTX fit lookup | Validation `source_id` and video stem equal the fitted checkpoint stem | Naming policy |

## Notes

- The derived validation clips contain 49 frames. With the trained 32-prefix/16-stride contract,
  each supplies one complete held-out continuation view and leaves the final frame outside the
  commit interval.
- Target topology remains privileged in this evaluation; the test isolates cross-domain mark
  realization before scaffold-conditioned occupancy/count generation.

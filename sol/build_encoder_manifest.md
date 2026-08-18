# `build_encoder_manifest.py`

## Purpose

Turns one or more generated-corpus manifests into an encoder training manifest. Because the
amortized encoder is supervised by render loss against the video alone — no fitted field, no
prompt, no caption — every completed clip is trainable the moment it lands, which lets encoder
training pipeline against a still-running harvest.

## Components

### `build_encoder_manifest`
- **Does**: Keeps only `status == "complete"` clips, maps the corpus's `evaluation` prompt role
  to the validation split (one held-out clip per class per style), tags every example with its
  style, and prefixes source IDs with the style so identical prompts across style passes stay
  distinguishable.
- **Rationale**: Refuses a manifest with no completed evaluation clip rather than silently
  training without a held-out set.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `train_amortized_encoder.py` | `source_id`, `split`, `video`, `frames`, `start_frame` per example | Field names |
| Stage 0 G1 reporting | `style` on every example so gates report per domain | Tag policy |

## Notes

- Re-runnable: rebuilding mid-harvest simply picks up the newly completed clips.

# `train_prompted_streaming.py`

## Purpose

Trains the first closed-vocabulary promptable jewel model directly on persistent continuation
births. It bypasses the noisy learned reconstruction tokenizer and keeps carried jewels exact.

## Components

### `PreparedPromptedView`
- **Does**: Keeps one shared-normalized local prefix raster, sparse birth target, and owned training
  prompt rows ready on the target device.

### `main`
- **Does**: Loads the leakage-safe 12-train/4-validation prompt corpus, cycles training views and
  prompt templates, trains direct jewel marks/counts, evaluates prompt controls, and saves complete
  provenance.
- **Text dropout**: Trains the learned null prompt required for classifier-free controls.
- **Context dropout**: Trains on zero-prefix examples so text-only generation cannot ignore the
  action prompt in favor of scene context.
- **Sparse count control**: `--balance-count-loss` gives occupied and empty birth cells equal
  objective weight so prompt-dependent topology is not trained mostly on zeros.
- **Interacts with**: `streaming_corpus.py`, `streaming_model.py`, and
  `prompted_streaming_eval.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Prompt gate | Group-4 videos and held-out prompt templates never train the model | Split ownership |
| Recovery | Checkpoint restores model, optimizer, scaler, and exact condition provenance | Save schema |
| Future rollout | Predicted births are frontier-local; carried jewels remain external and exact | Output semantics |

## Notes

- This first model generates a future stride, not an initial full video. Context dropout establishes
  the text-only semantic path needed before adding a learned initial-state prior.
- Oracle-rank prompt metrics are a preflight. Free decoded counts and rendered videos remain the
  decisive next gate.

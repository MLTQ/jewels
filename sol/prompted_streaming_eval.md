# `prompted_streaming_eval.py`

## Purpose

Measures whether direct jewel continuation uses the requested action text on wholly held-out source
groups. It evaluates both ordinary prefix continuation and text-only generation with the prefix
removed.

## Components

### `evaluate_prompted_streaming`
- **Does**: Holds target birth ranks fixed while comparing the correct unseen prompt template, a
  deterministic different-class prompt, and the trained null condition.
- **Does**: Repeats those controls with full local prefix context and with a zeroed prefix raster.
- **Interacts with**: `PromptedContinuationCorpus` and `BirthContinuationModel`.
- **Rationale**: A strong prefix can reveal the action and hide text neglect; text-only controls
  force the prompt to carry semantic information.

### `PromptedStreamingEvaluation` / `PromptControlMetrics`
- **Does**: Report standardized birth-feature MSE and per-cell birth-count MAE for every control.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Prompt trainer | Correct/shuffled/null use identical ranks and weights | Control construction |
| Research gate | Shuffled prompts always come from a different class | Class rotation |
| Future visual audit | Text-only correctness must pass before expensive free renders | Metric schema |

## Notes

- Oracle ranks isolate semantic mark prediction from free-running topology. Independently decoded
  videos remain a subsequent gate.

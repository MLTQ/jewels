# `streaming_continuation_eval.py`

## Purpose

Evaluates whether the learned sparse future actually depends on the correct prefix and whether its
predicted birth marks reconstruct the target future field while carried jewels remain exact.

## Components

### `ContinuationEvaluation`

- **Does**: records correct/shuffled/null feature, count, and rendered-field metrics plus predicted
  count ratio and carried-state error

### `evaluate_continuation`

- **Does**: holds target ranks and model weights fixed while swapping only context embeddings;
  selects a shuffled context whose source interval is disjoint from the evaluated target stride,
  returns local predictions to global time, and compares sampled renders
- **Interacts with**: `streaming_model.py`, `streaming_features.py`, and `render.py`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Overfit gate | Correct prefix beats shuffled and null on identical future targets | Control construction |
| Persistent state | Candidate field concatenates exact carried jewels with predicted births | Merge semantics |
| Prompt successor | Same controls can later swap text conditions as well as prefixes | Metric fields |

## Notes

- Render PSNR compares predicted and target jewel fields, not the source pixels; stage-1
  reconstruction quality is reported separately.
- Feature-render evaluation uses oracle target birth counts/ranks to isolate mark prediction. The
  independently decoded count ratio reports the variable-count path.
- Shuffled controls never draw from a prefix that overlaps the target stride. This prevents future
  leakage when adjacent streaming prefixes overlap one another.
- A future stride with no births contributes zero mark error while still contributing count and
  carried-field render metrics.

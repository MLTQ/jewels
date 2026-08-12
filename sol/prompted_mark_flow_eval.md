# `prompted_mark_flow_eval.py`

## Purpose

Measures whether the stochastic birth-mark velocity field uses prefix and prompt conditions on
held-out source groups. It fixes noise, flow time, and oracle topology across controls.

## Components

### `evaluate_prompted_mark_flow`
- **Does**: Scores correct, different-class, and trained-null text under full-prefix and text-only
  contexts for every held-out continuation view, with an optional matched video-guide raster or
  multiscale token set.
- **Interacts with**: `birth_mark_flow_objective` and the leakage-safe prompt corpus.
- **Rationale**: Sharing each stochastic flow path prevents noise variation from masquerading as
  semantic selectivity.

### `PromptedMarkFlowEvaluation` / `MarkFlowControls`
- **Does**: Provide a stable JSON-ready held-out flow-MSE schema.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Mark-flow trainer | Evaluation uses validation sources and held-out prompt templates only | Split policy |
| Research gate | Shuffled text always belongs to a different action class | Class rotation |
| Visual sampler | Metrics hold topology fixed and therefore assess marks, not counts | Interpretation |
| Oracle-guide experiment | Every guided validation view supplies its own aligned RGB raster | Guide mapping |
| Multiscale experiment | Every guided validation view supplies its own `(cell,token,feature)` set | Token mapping |

## Notes

- Fixed-path MSE is a conditioning diagnostic. Sampled renders remain the quality gate.
- Empty-birth strides do not define a mark velocity and are excluded from the reported view count;
  they belong to the future topology objective.

# `scaffold_mark_eval.py`

## Purpose

Measures whether the shared 1,024-rank flow uses the video scaffold in both the empty-state initial
window and later generated-state-compatible continuation windows.

## Components

### `evaluate_scaffold_mark_flow`

- **Does**: Scores fixed noise/time paths under correct, null-scaffold, and no-context conditions
  on held-out sources. It also reports a cross-class-shuffled condition whenever at least two
  validation classes exist.
- **Interacts with**: `birth_mark_flow_objective` and `scaffold_mark_data.py`.
- **Rationale**: Holding target topology and stochastic paths fixed attributes loss differences to
  conditioning rather than count or sampling variation.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Scaffold mark trainer | Evaluation separates initial and continuation rows | Report schema |
| Research gate | Shuffled guide is from a different held-out action class | Control mapping |
| Visual rollout | Correct/null/shuffled controls share the same mark model | Conditioning ownership |

## Notes

- Fixed-path MSE is a conditioning diagnostic. Autonomous multi-window renders remain the gate.
- A single-field memorization corpus cannot define an honest cross-class shuffle. Its report sets
  `shuffled_scaffold_available` false and omits only the shuffled loss/delta; correct, null, and
  no-context controls remain comparable.

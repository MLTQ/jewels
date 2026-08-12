# `scaffold_mark_eval.py`

## Purpose

Measures whether the shared 1,024-rank flow uses the video scaffold in both the empty-state initial
window and later generated-state-compatible continuation windows.

## Components

### `evaluate_scaffold_mark_flow`

- **Does**: Scores fixed noise/time paths under correct, cross-class-shuffled, null-scaffold, and
  no-context conditions on held-out sources.
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

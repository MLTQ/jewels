# `train_masked_axial_prior.py`

## Purpose

Converts the full-generation axial prior into an editor-aware repair model by fine-tuning on the
same state distribution used at inference: clean context at its target value and noisy cuboid holes.

## Method

- Start from the full prior's EMA weights.
- Add a zero-initialized clean/dirty cell embedding so initialization exactly matches the base
  velocity field while making the repair request explicit.
- Sample random coarse cuboids approximating swept parallelepiped edits.
- Interpolate noise to target only inside each cuboid and score velocity only there.
- Retain a small full-generation flow loss to reduce catastrophic forgetting.
- Track an EMA and compare held-out dirty-region MSE against both the frozen base model and
  normalized zero filling.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Hierarchical editor | Checkpoint uses the axial callable plus optional per-cell mask | Architecture metadata |
| Experiment audit | Base and latest repair metrics share fixed masks and noise | Evaluation seed/protocol |
| Full generator | Full-loss regularizer remains explicit in checkpoint arguments | Objective mixture |

## Notes

This stage teaches context-based hole repair, not protected-jewel collision handling. The latter
needs a set or per-cell encoder for moved jewels in the prior condition.

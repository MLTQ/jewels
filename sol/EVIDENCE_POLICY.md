# Experimental evidence policy

This project contains useful experiment logs, but their prose conclusions are not all equally
supported. Apply this policy when choosing work or writing new reports.

## Asymmetric interpretation

- A success is an existence proof under the recorded conditions. It establishes that the mechanism
  can work, though not that it generalizes or is the best mechanism.
- A failure is only a negative observation under the recorded implementation, data, seed, compute,
  and hyperparameters. It does not establish that the mechanism cannot work.
- Statements such as “settled,” “cannot,” “no amount of capacity,” “do not revisit,” or “objective
  law” are not inherited from old reports unless the underlying evidence satisfies the replication
  requirements below.

## Labels for new conclusions

1. **Observed** — one completed configuration.
2. **Replicated** — at least three seeds and three meaningfully different clips or datasets.
3. **Causal signal** — matched intervention/control with the intended variable isolated.
4. **Scaling signal** — a monotonic or fitted trend across at least three compute or data budgets.
5. **Decision-grade** — replicated causal/scaling evidence, implementation checks, and uncertainty
   reported. This can prioritize work; it is still not a universal law.

Every new report should use the strongest label actually earned and name the remaining
generalization gap.

## Requirements before retiring an idea after failure

- Verify the implementation with a positive control, oracle, or constructed case.
- Match training data, parameter budget, optimizer opportunity, evaluation renderer, and metric.
- Sweep enough optimization compute to distinguish a bad mechanism from undertraining.
- Run at least three seeds and three clips/datasets representative of the target domain.
- Report individual runs and dispersion, not only a macro average.
- State the bounded conclusion: which implementation failed, under which conditions.

Ideas that do not meet these requirements remain eligible for targeted re-examination. This does
not mean rerunning every old arm; prioritize mechanisms whose success would remove a current
feasibility blocker.

## Current branch

The support-correct scaling and temporal-tilt experiments on
`codex/support-correct-scaling-proof` are initial scaling and causal signals. Single-source results
must be replicated before the text-to-video feasibility pitch.

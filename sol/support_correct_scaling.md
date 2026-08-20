# `support_correct_scaling.py`

## Purpose

Runs the first evidence-bearing stage-1 experiment after discovering that production
center-distance KNN can omit elongated spacetime splats. It separates renderer correctness from
the question of whether more fitting compute improves the representation.

Interpret conclusions under `sol/EVIDENCE_POLICY.md`: a single-source curve is a scaling signal,
not a universal claim about the representation.

## Experiment

- Fits the same clip, initialization seed, primitive budget, and sampled voxels with legacy KNN,
  the all-center conservative finite-support oracle, or the support-complete tiled implementation.
- Repeats each arm across optimizer-step budgets.
- Evaluates every field with the support-correct renderer, even when it was trained with KNN.
- Also reports the pixel gap between the training renderer and the support renderer.
- Records anisotropy, longest-axis temporal alignment, mixed spacetime tilt, five-sigma temporal
  lifespan, opacity, primitive count, wall time, and full-volume PSNR. Mixed tilt is zero for a
  purely spatial or purely temporal axis and one when its spatial/time components are balanced.

The script writes a checkpoint for every arm plus an incrementally updated `report.json`, so an
interrupted matrix retains completed evidence.

## Initial proof gate

The report marks four deliberately modest checks: a support arm exists, its PSNR improves by more
than 0.5 dB along the requested compute curve, the largest support-safe run reaches 25 dB, and the
median fitted anisotropy exceeds 1.5. Passing is not by itself proof of promptable text-to-video;
it says the corrected stage-1 substrate responds to compute without collapsing to isotropic dots.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment reports | `support-correct-scaling-v1` with per-run records and summary | Schema or metric definitions |
| Stage-1 checkpoints | `state`, `cfg`, and `info` keys | Fitter checkpoint format |

## Notes

- `--cull-mode exact` is intentionally absent from the matrix. It is an oracle for tiny unit tests,
  not a practical training arm.
- Candidate overflow from either support mode is allowed to terminate the run. That is a measured
  capacity failure, not an error the experiment should hide. Tiled resolution and level spacing are
  stored in each checkpoint.

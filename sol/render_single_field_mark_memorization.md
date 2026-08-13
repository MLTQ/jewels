# Single-field mark memorization renderer

## Intent

`render_single_field_mark_memorization.py` is the visual gate between the
current scaffold reconstruction experiments and larger prompt-conditioned
training. It compares two mark-flow checkpoints trained for the same total
number of updates on one physical fitted field: a feature-only continuation
control and a render-supervised branch with one learned RGB background.

Every branch receives fitted birth cells and ranks (exact topology), and every
continuation window receives fitted carried jewels (teacher-forced carry). A
failure in this audit is therefore a mark/background failure, not a topology or
generated-state failure.

## Inputs and ownership

The manifest must have been reduced by `build_ltx_style_train.py --class-name`.
It contains exactly one training alias and one validation alias for the same
`shared_field_stem`. Both checkpoints must own the prompt-cache digest, grid,
normalizers, and architecture. The rendered checkpoint must declare
`single_field_learned_rgb` and contain its learned background.

The validation alias supplies the evaluation prompt and video guide. The paired
aliases intentionally refer to the same field: this is a memorization/capacity
test, not a held-out generalization estimate.

## Outputs

The animation and contact sheet show the source video, fitted-jewel ceiling,
feature-only marks with fitted background, render-supervised marks with fitted
background, and render-supervised marks with learned background.

`summary.json` reports ordinary render signatures, saliency/motion signatures,
window seams, and teacher-forced per-frame splat density. Metrics are reported
against both the source and fitted ceiling. The former measures the complete
jewel representation; the latter measures whether the generator memorized the
representation it was trained to generate.

`--deterministic` enables repeatable CUDA kernels before model construction and
records that contract in the summary. Final research artifacts use this flag;
without it, device/kernel differences can move aggregate scores slightly even
when the Gaussian sampling seed is fixed.

## Interpretation

- Equal failure at matched update count means more of the same mark training is
  not the immediate remedy.
- Closing the gap to the fitted ceiling with render supervision identifies the
  objective as the bottleneck and supports testing generated carry next.
- A poor fitted ceiling cannot be repaired by generator scaling; fitting or the
  representation must improve first.
- A learned-background gain over the fitted-background twin identifies a
  remaining field/background partition problem.

The audit does not claim autonomous generation quality because topology and
carry are intentionally supplied from the target.

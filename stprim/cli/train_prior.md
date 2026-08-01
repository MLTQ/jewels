# train_prior.py

## Purpose
Train the v0 set prior on a fitted corpus with rectified-flow matching.

## Components

### `main()`
- **Does**: load corpus -> standardize features (per-dim mean/std over all S·N primitives) ->
  flow-matching loop (x_t = (1-t)x0 + t·x1, target v = x1 - x0, MSE) with CLIP conditioning
  dropout -> save model + meta (stats, bg_mean, shape, model_args) to `prior.pt`
- **Interacts with**: `prior/featurize.py`, `prior/model.py`

## Decisions
- Normalization stats live INSIDE the checkpoint meta — a prior is meaningless without the
  exact stats it was trained under, so they travel together.
- `bg_mean` (corpus-mean additive background) is a v0 simplification: the background is
  per-set in stage 1 but nearly constant on a single-scene corpus. A set-level head (or a
  background token) is the upgrade when corpora get diverse.
- Memorization on 231 sets is intended, not a bug: v0 validates the mechanics and gives the
  round-trip upper bound. Generalization questions start with bigger corpora.
- **v1 additions (2026-07-31):** EMA weights (saved as `ema` in the checkpoint; the sampler
  prefers them), bf16 autocast (default on, `--no-bf16`), warmup+cosine LR, and `--flip-u`
  mirror augmentation done in RAW feature space — negate mu_u, the uv/ut log-covariance
  entries, and the d/du column of the P1 ramp (congruence by diag(-1,1,1); log commutes with
  orthogonal congruence). CLIP embeddings are reused for flips (CLIP is ~mirror-invariant).

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `cli/sample_prior.py` | `prior.pt` = {model, meta{feat_mean, feat_std, bg_mean, shape, n_primitives, model_args}, step} | meta schema |

## Notes
- ~0.3 s/step at v0 scale -> a 6000-step run is ~30 min on the 4090.

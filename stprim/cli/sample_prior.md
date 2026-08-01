# sample_prior.py

## Purpose
The moment of truth: sample sets from the prior, decode to PrimitiveFields, render video.

## Components

### `sample_sets(...)`
- **Does**: Euler integration of the learned velocity from noise, optional classifier-free
  guidance (`v_u + cfg·(v_c - v_u)`) toward a CLIP embedding

### `main()`
- **Does**: per sample -> un-normalize with the checkpoint's stats -> `features_to_field` ->
  full-volume render with the corpus-mean background -> GIF as [real fitted window | sample];
  prints real-vs-generated marginal stats as a cheap distribution check
- **Interacts with**: `prior/featurize.py`, `prior/model.py`, `models/render.py`,
  montage helpers imported from `cli/render_recon.py`

## Decisions
- Every sample GIF carries the real fitted reference in the left panel — a generated video is
  uninterpretable without an anchor for what "a fit of this scene" looks like.
- Conditioning comes from a corpus window's CLIP sidecar (`--cond-index`), because on a
  single-scene corpus there is no meaningful text prompt yet; `--cond-index -1` samples
  unconditionally. Text prompts arrive with a diverse corpus, through the same interface.

## Notes
- Samples EMA weights when the checkpoint has them (`--raw-weights` to override).
- Montage helpers are imported from `cli/render_recon.py`; if viz helpers grow further they
  should move to their own module.

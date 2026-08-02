# eval_prior.py

## Purpose
One comparable number per prior checkpoint, so scaling claims are a measured curve instead of
three GIFs and adjectives. Metric: Fréchet distance in CLIP ViT-B/32 image-embedding space
between generated renders and fitted renders ("CFD").

## Components

### `frechet(a, b)`
- **Does**: Fréchet/FID formula in double precision; Tr√(CaCb) via the symmetric sandwich
  (eigh-only — no scipy on the training box)

### `main()`
- **Does**: fixed sampling protocol (seed 0, cfg 1.5, 100 flow steps, cond-index 0) -> render
  -> CLIP-embed -> CFD against a CACHED reference embedding set
- **Interacts with**: `prior/featurize`, `prior/model`, `cli/sample_prior.sample_sets`

## Decisions
- **Generated renders vs FITTED renders**, not raw video: isolates prior quality from fitter
  softness. The fitter's own gap to pixels is measured separately (PSNR, stage 1).
- **Reference embeddings cached** in the corpus dir: every checkpoint ever scored uses the
  byte-identical reference — deleting `real_clip_stats.pt` invalidates the whole curve.
- Small-sample Fréchet is biased; the bias is constant under a fixed protocol, so the CURVE
  is meaningful even where absolute values are not comparable to published FID/FVD numbers.
  (Never quote CFD against another paper's FVD.)

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| scaling-curve claims (README/paper) | protocol constants above stay frozen | any change to sampling params, frame picks, CLIP model, or reference cache resets the curve |

## Notes
- Runs fine on the second GPU (cuda:1) while the 4090 trains — sampling small models + CLIP
  fits in 8 GB with the chunked culler.

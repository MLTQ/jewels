# Prompt-conditioned native Jewel caster Gate 1 protocol

## Decision question

Can text directly parameterize the passing one-Jewel language without a source video, dense video
latent, target centroids, or target token IDs at inference?

This is the first promptability proof, not a production text-to-video model. It models a Jewel field
as a text-conditioned marked point process: a continuous centroid density casts irregular points;
a coordinate-aware token head speaks covariance, surface/opacity, and gradient vocabulary marks at
those points.

## Frozen data split and prompts

- Train: the same 33 source-disjoint optimized fields used by Gate 0d/0f.
- Validation: cartoon dog park, 3D workshop, and anime ballet, each at fitter seeds 0/1/2.
- The held-out style/action combinations are absent from training, while the action sentence appears
  in other styles and each style appears with other actions. This is a compositional prompt holdout.
- Conditioning text is `"{style} video. {source_prompt}"` from checkpoint metadata.
- Frozen local text encoder: `BAAI/bge-small-en-v1.5`; no fine-tuning.

## Frozen model and training

- Freeze the passing source-disjoint Gate-0d bundle-1 K=1,024 codebook.
- Sample 8,192 Jewels per training source and 16,384 per validation field, deterministic seed
  20260831. Only sampled train Jewels update weights.
- Density head: 64-component diagonal Gaussian mixture predicted from the frozen 384D text vector.
- Token head: text vector plus four-frequency continuous centroid features, four 512-wide SiLU
  layers, and three 1,024-way active-role logits.
- Prompt dropout 10% trains an explicit null-prompt control.
- Joint loss: mean active-role cross-entropy plus 0.1 times centroid negative log-likelihood.
- AdamW 3e-4, weight decay 1e-4, batch 4,096, maximum 15,000 updates; evaluate every 500 and stop
  after ten evaluations without 0.1% relative true-prompt validation improvement.
- Retain the best true-prompt validation checkpoint.

## Frozen controls and free run

- `correct`: held-out style plus its action sentence.
- `shuffled`: cyclically exchange the three held-out prompts while keeping target Jewels fixed.
- `null`: empty text.
- Teacher-forced evaluation supplies target centroids only to measure text-conditional density and
  token likelihood; it is not the generation result.
- Free run samples 72,000 continuous centroids and all three token IDs from text alone. Correct,
  shuffled, and null arms share declared seeds. Fields decode directly through the frozen Jewel
  vocabulary and support renderer.

## Registered Gate 1

All checks must pass:

1. Correct-prompt centroid NLL is below both shuffled and null controls.
2. Correct-prompt token NLL is below both controls for all three roles; macro improvement over the
   better control is at least 2%.
3. A correct-prompt free-run field's cell-conditional active-token histogram is closer to its
   matching held-out field than shuffled/null generations by at least 0.02 cosine.
4. Correct-prompt free-run random-volume PSNR exceeds both shuffled and null controls by at least
   0.25 dB on average.
5. Generated centroids have less than 1% exact cell-center locking.
6. Inference audit confirms no target field, centroid, token, pixels, video latent, class ID, or
   source identity is an input; only text and declared random seed are used.

A pass is the pitch milestone: native text-to-Jewel-video generation has a controlled proof, and
data/compute scaling is the next claim. A failure distinguishes density, token semantics, or visual
composition and blocks the compute pitch until that component is repaired.

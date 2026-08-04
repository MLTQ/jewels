# Latent-prior spike artifacts

This directory preserves selected metrics and source-balanced visual comparisons from the first
text-conditioned raster-flow experiments.

## Decision

The raster flow passes the latent-distribution and conditioning-plumbing spike, but fine video
detail is currently bounded by the tokenizer. Improve the deterministic bottleneck before spending
more compute on the prior.

## Protocol

- Frozen tokenizer: 5,000-step render-aware checkpoint from `../README.md`
- Latents: 864 raster cells × 32 dimensions, normalized per cell from 214 training windows
- Validation: all 17 windows from wholly held-out sources `11` and `15`
- Condition: 512-D CLIP ViT-B/32 mean image embedding
- Prior: 7.73M-parameter six-layer rectified flow, batch two, fp16, EMA
- GPU: RTX 2070 SUPER (`cuda:1`, 8 GB)

## Conditioning ablation

Raw unit CLIP conditions were effectively ignored after 10,000 steps: correct-condition held-out
flow MSE was 0.86693 versus 0.86672 with shuffled conditions. Per-dimension whitening using training
sources only made the condition measurable after 5,000 steps:

| Held-out flow path | MSE |
|---|---:|
| Correct whitened CLIP | 0.92486 |
| Shuffled whitened CLIP | 0.93010 |
| Unconditional | 0.93463 |

The semantic margin is real but small. This corpus proves that the condition reaches the velocity
field; its single surveillance domain and image-text modality gap cannot prove broad prompting.

## Distribution gate

Paired generated-target MSE was rejected as a primary stochastic metric because it rewards a
collapsed mean over independent samples. With 17 generated and 17 held-out normalized latent fields,
empirical energy distance was:

| Distribution | Energy distance (lower is better) |
|---|---:|
| Conditional prior, CFG 1.0 | **0.23648** |
| Conditional prior, CFG 1.5 | 0.24150 |
| CLIP nearest training window | 0.31272 |
| Repeated scene-mean latent | 0.78905 |

The learned flow therefore beats both non-generative baselines and does not collapse to the mean.
CFG 1.0 is preferred because the semantic margin is not strong enough to benefit from guidance.

## Visual gate

[`11_w000000_comparison.gif`](11_w000000_comparison.gif) and
[`15_w000000_comparison.gif`](15_w000000_comparison.gif) compare held-out fitted target, tokenizer
round-trip, and conditional prior sample over 16 temporal positions. Jewel counts stay stable:
generated samples range from 6,253 to 6,593 across all held-out conditions around the 6,471 target.

The prior reproduces the broad fixed-camera scene, palette, and smooth temporal field. However,
people and other small foreground structures are already heavily suppressed in the tokenizer
round-trip. This makes tokenizer detail—not prior capacity—the next limiting factor.

## Artifacts

- `whitened_eval_cfg1.json`: selected all-window evaluation
- `whitened_eval_cfg15.json`: matched guidance comparison
- `raw_condition_eval_cfg1.json`: pre-whitening control
- `whitened_train_log.jsonl` / `whitened_summary.json`: selected training curve
- `visual_manifest.json`: exact windows, temporal indices, and jewel counts

The selected prior checkpoint remains on the training host at
`/home/m/jewels/prior/sol_flow_whitened_5k/prior.pt`.

# PlayingGuitar single-field memorization gate

This experiment answers whether the noisy cel generator primarily needs more
optimization/data, or whether its mark objective and lifecycle parameterization
are wrong. The result is decisive: **the field is not too sparse, and strong
whole-mark render supervision is actively harmful**. A weak rendered loss is a
useful spatial regularizer, but late temporal noise remains and selects the
frozen-lifecycle, appearance-only adapter tracked by `jewels-brv`.

## Scope and privilege contract

The 49-frame PlayingGuitar LTX clip is represented by one 72k fitted jewel field.
The manifest contains one training alias and one evaluation-prompt alias for the
same `shared_field_stem`; this is explicitly a memorization/capacity result, not
unseen-prompt or unseen-video generalization.

Every rendered comparison supplies exact fitted birth cells/ranks and fitted
carried jewels to both models. The generated quantity is only each stride's
22-dimensional jewel marks. Consequently, visual failure here cannot be blamed
on predicted topology or autoregressive carry drift. Three complete 16-frame
strides produce the 48-frame audit.

The fitted ceiling averages 6,189 effective and 7,503 5%-visible splats/frame,
squarely inside the requested 5k--10k regime. Its source-relative reconstruction
is 18.811 dB / 0.8667 SSIM at the audit resolution.

## Matched training

All branches start from the four-field cel mark-flow checkpoint, then share a
5,000-step feature-only single-field base. Text, context, and guide dropout are
zero. The feature control and rendered branches each receive another 5,000
updates from that exact base and use seed 41.

| Stage | GPU | Updates | Runtime | Objective |
|---|---|---:|---:|---|
| Shared feature base | RTX 2070 SUPER | 5,000 | 61.73 s | feature MSE |
| Feature control | RTX 2070 SUPER | +5,000 | 65.28 s | feature MSE |
| Strong rendered control | RTX 4090 | +5,000 | 438.01 s | 0.25 feature + 2.0 rendered + 0.25 frontier |
| Conservative rendered branch | RTX 4090 | +5,000 | 496.86 s | 1.0 feature + 0.02 rendered, no frontier |

The strong arm uses foreground/motion/stability components, frontier anchoring,
and a contribution loss. The conservative arm uses four four-frame 6x6 patches,
half scaffold-salient sampling, and low-weight RGB/edge/chroma/structure/motion/
stability terms. Both learn one sigmoid-bounded RGB background initialized only
from the first causal guide stride.

An initial smoke test uncovered a previously hidden implementation failure:
mixed-precision loss scaling through the differentiable covariance renderer
produced infinite/NaN gradients even though forward losses were finite. The
scaler therefore skipped rendered updates. The trainer now automatically uses
full-precision backpropagation whenever render supervision is active. All
reported rendered runs have finite gradients; feature-only training retains AMP.

## Strong rendered objective: rejected

The strong objective lowers its sampled patch loss while destroying the global
field. Deterministic 20-step sampling on the same 2070S gives:

| Against source | Feature control | Strong rendered |
|---|---:|---:|
| PSNR | 17.426 dB | 13.723 dB |
| SSIM | 0.7873 | 0.7620 |
| Foreground RGB MAE | 0.1591 | 0.3505 |
| Quiet temporal MAE | 0.03685 | 0.15785 |
| Seam / target seam | 103.98x | 828.15x |
| Effective splats/frame | 6,258 | 7,933 |
| 5%-visible splats/frame | 7,496 | 9,040 |

The contact sheet shows the mechanism: each stride begins recognizably, then
ends in saturated red/green masses. Render pressure plus the frontier/lifecycle
term inflates contribution and temporal support rather than improving detail.
This is why adding more jewels or more steps is not the remedy.

The learned background converges within 0.0043 mean RGB of the fitted background,
yet its panel is no better. Background estimation is therefore not the material
bottleneck.

## Conservative objective: bounded spatial gain

At the matched 5,000-step endpoint on the 4090, conservative render supervision
keeps density unchanged and improves spatial structure relative to the feature
control evaluated in the same deterministic invocation:

| Metric | Feature control | Conservative | Change |
|---|---:|---:|---:|
| Source PSNR | 17.471 dB | 17.471 dB | +0.001 |
| Source SSIM | 0.7870 | 0.7925 | +0.0055 |
| Source contrast ratio | 0.9504 | 0.9699 | toward 1.216 ceiling |
| Source edge ratio | 1.6430 | 1.6659 | toward 1.746 ceiling |
| Fitted-ceiling PSNR | 15.428 dB | 15.538 dB | +0.110 |
| Fitted-ceiling SSIM | 0.7335 | 0.7458 | +0.0122 |
| Fitted foreground RGB MAE | 0.2200 | 0.2156 | -0.0044 |
| Fitted quiet temporal MAE | 0.03165 | 0.03201 | +0.00035 (worse) |
| Effective splats/frame | 6,277 | 6,279 | unchanged |

The preserved 3,000-step early stop is better balanced. On the 2070S it raises
source PSNR 17.426 to 17.511 and SSIM 0.7873 to 0.7951, lowers source foreground
MAE 0.1591 to 0.1567 and quiet error 0.03685 to 0.03674, and raises fitted-ceiling
PSNR 15.416 to 15.538 while density changes from 6,258 to 6,255. The final branch
retains the spatial gain but begins to trade away temporal stability.

The learned-background panel is again slightly worse than using the fitted
background. This is diagnostic only: the learned RGB proves the background can
be estimated causally, but moving that scalar gauge does not solve jewel motion.

## Decision

The idea remains promising. Under exact topology and fitted carry, ordinary mark
training produces a recognizable field only 1.34 dB below the fitted ceiling.
The remaining artifact has a repeatable temporal shape: stride starts are good
and stride ends dissolve. This localizes the next work to lifecycle/covariance
and appearance coupling inside the mark generator.

Do not expand the corpus yet. More fields are necessary later for prompt
generalization, but they would currently teach around a broken objective. The
next gate is `jewels-brv`: freeze the proven base, topology, lifecycle dimensions,
density, and stable IDs; train a small zero-initialized appearance residual first
on RGB in scaffold-salient cells; expand dimensions only if spatial metrics
improve with no quiet/motion regression. After that passes on this field, repeat
under generated carry, then expand the styled corpus.

## Artifacts

- `manifest.json` and `prompts.pt`: exact single-field ownership and prompt cache.
- `training/`: compact summaries and full training curves for all four stages.
- `strong_render_5000/`: deterministic rejected-control GIF, contact sheet, and
  full metric/density report.
- `conservative_render_5000/`: deterministic matched-step endpoint.
- `conservative_render_step3000/`: preserved selected early stop.
- `interim_1000/`: early strong-objective evidence retained for trajectory audit.

Remote recovery root:
`/home/m/jewels/topology/ltx_cel_single_guitar_v1`. Checkpoints are intentionally
kept on Aine rather than committed to Git.

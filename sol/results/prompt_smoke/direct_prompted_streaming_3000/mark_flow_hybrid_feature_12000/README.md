# Hybrid raster-plus-token guide control

This 12,000-step control isolates the multiscale conditioning architecture from the render loss.
It retains the complete v1 cell-RGB 3D-convolution/global guide path, adds within-cell multiscale
token cross-attention residually, and trains with feature-space rectified-flow loss only. The data,
held-out group, seed, target topology, sampler, and renderer match the selected v1 oracle-guide run.

## Result: rejected

| Guided projected mean | v1 cell-RGB guide | Hybrid raster + tokens | Change |
|---|---:|---:|---:|
| PSNR | 16.555 dB | 16.240 dB | -0.316 dB |
| SSIM | 0.8475 | 0.8345 | -0.0130 |
| Target contrast | 0.8555 | 0.8519 | -0.0037 |
| Target edge energy | 0.9050 | 0.8925 | -0.0125 |
| Target saturation | 0.9371 | 0.9209 | -0.0162 |
| Target temporal change | 1.7236 | 1.7250 | +0.0014 overshoot |

Mean full birth-cell adherence also falls from 0.9932 to 0.9849. Basketball gains 0.0040 SSIM
and modest contrast/edge energy, but the other three classes do not reproduce that improvement;
ApplyEyeMakeup loses 0.795 dB and HorseRiding loses 0.0287 SSIM. The contact sheets agree with the
paired metrics: local tokens do not add reliable detail and can destabilize moving foregrounds.

This result rejects the tested local-token residual at the current data and capacity. It does not
reject multiscale video evidence in general, but it makes the original v1 raster guide the selected
conditioner. The next controlled experiment keeps that architecture fixed and adds only the
differentiable RGB/edge/chroma/SSIM render objective.

- `summary.json` and `train.log` record the 553.8-second training run.
- `v1_comparison.json` contains matched macro and per-class signed deltas.
- `visual_contract_projection/mark_flow_visual_report.json` contains all per-class metrics.
- The GIFs/contact sheets use the same target, carry, deterministic, zero-guide, guided,
  projection, and shuffled-text panels as v1.

# Raster-guide plus render-loss control (weight 0.5)

This final 12,000-step objective control keeps the selected v1 cell-RGB conditioner and adds the
same sampled RGB/edge/chroma/SSIM render loss as the weight-2 run, but at mean weight 0.5. That
value was selected before seeing these results because it won the project's earlier tokenizer
render-loss sweep. Data, held-out group, seed, capacity, topology, sampler, and renderer match v1.

## Result: useful trade, not selected

| Guided projected mean | v1 feature flow | Render weight 0.5 | Change |
|---|---:|---:|---:|
| PSNR | 16.555 dB | 16.248 dB | -0.307 dB |
| SSIM | 0.8475 | 0.8557 | +0.0082 |
| Target contrast | 0.8555 | 0.9351 | +0.0795 |
| Target edge energy | 0.9050 | 0.9251 | +0.0200 |
| Target saturation | 0.9371 | 0.9167 | -0.0204 |
| Target temporal change | 1.7236 | 1.8055 | +0.0818 overshoot |

ApplyEyeMakeup is a real per-class win (+0.584 dB, +0.0237 SSIM, and higher edge energy), while
Basketball is nearly flat and HorseRiding is mixed. PlayingGuitar is unstable: it loses 1.574 dB,
overshoots target contrast to 1.075, and adds 0.278 target-relative temporal change. The aggregate
SSIM/detail improvement therefore does not meet the requirement for consistent four-class
fidelity and temporal behavior.

No further scalar-weight tuning is warranted. The result says the differentiable objective carries
useful detail information, but its current uniform tiny-patch estimator and temporal weighting can
amplify motion inconsistently. The v1 feature-flow model remains selected for the LTX transfer gate;
a future render objective should use foreground/motion-aware sampling and an explicit temporal
stability term.

- `summary.json` and `train_log.jsonl` record the 263.7-second training run.
- `v1_comparison.json` contains matched macro and per-class signed deltas.
- `visual_contract_projection/mark_flow_visual_report.json` contains all per-class metrics.
- The GIFs/contact sheets use the same visual contract as v1.

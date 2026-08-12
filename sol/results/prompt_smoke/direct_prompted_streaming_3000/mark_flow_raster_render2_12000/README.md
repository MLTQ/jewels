# Raster-guide plus render-loss control (weight 2)

This 12,000-step control keeps the selected v1 cell-RGB 3D-convolution/global conditioner and
changes only the training objective. Every fourth feature-flow update also supervises the
denoised, topology-projected marks through sampled RGB, spatial/temporal edge, opponent-chroma,
and SSIM render losses. Cadence correction makes the mean render weight 2.0. Data, held-out group,
seed, model capacity, target topology, sampler, and renderer match v1.

## Result: rejected

| Guided projected mean | v1 feature flow | Render weight 2 | Change |
|---|---:|---:|---:|
| PSNR | 16.555 dB | 16.009 dB | -0.547 dB |
| SSIM | 0.8475 | 0.8388 | -0.0087 |
| Target contrast | 0.8555 | 0.8936 | +0.0381 |
| Target edge energy | 0.9050 | 0.9021 | -0.0029 |
| Target saturation | 0.9371 | 0.8836 | -0.0535 |
| Target temporal change | 1.7236 | 1.7725 | +0.0488 overshoot |

The visual objective increases contrast but does not improve action geometry or aggregate edge
energy. It lowers paired PSNR and SSIM, desaturates the output, and increases already-excessive
temporal change. ApplyEyeMakeup and PlayingGuitar gain roughly 0.002 SSIM individually, but all
four classes lose PSNR and the other tradeoffs are inconsistent. This does not pass the realizer
gate.

One final bounded weight-0.5 control is licensed by the earlier tokenizer sweep. If it does not
improve the matched gate, the render objective should be redesigned rather than tuned further.

- `summary.json` and `train_log.jsonl` record the 264.4-second training run.
- `v1_comparison.json` contains matched macro and per-class signed deltas.
- `visual_contract_projection/mark_flow_visual_report.json` contains all per-class metrics.
- The GIFs/contact sheets use the same visual contract as the selected v1 run.

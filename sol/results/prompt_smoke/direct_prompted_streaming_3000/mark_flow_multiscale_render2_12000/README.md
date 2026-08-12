# Multiscale token-guide plus render-loss control

This 12,000-step run is the first implementation of within-cell multiscale video tokens,
cell/rank cross-attention, and denoised-endpoint RGB/edge/chroma/SSIM supervision. It uses exact
target topology and the same 12-train/4-group-held-out UCF protocol, noise seeds, renderer, and
20-step sampler as the v1 cell-RGB oracle guide.

## Result: informative, but not selected

| Guided projected mean | v1 cell-RGB guide | Token guide + render loss | Change |
|---|---:|---:|---:|
| PSNR | 16.555 dB | 16.419 dB | -0.137 dB |
| SSIM | 0.8475 | 0.8441 | -0.0033 |
| Target contrast | 0.8555 | 0.8694 | +0.0138 |
| Target edge energy | 0.9050 | 0.9108 | +0.0058 |
| Target saturation | 0.9371 | 0.9194 | -0.0177 |
| Target temporal change | 1.7236 | 1.7833 | +0.0597 overshoot |

Every projected sample remains in its declared spatial cell; mean full birth-cell adherence is
0.9886. The run improves local contrast and edge energy, but loses paired PSNR, SSIM, saturation,
and temporal stability. Per-class behavior is mixed: ApplyEyeMakeup improves PSNR/SSIM, HorseRiding
improves PSNR but loses SSIM, Basketball is nearly flat, and PlayingGuitar loses 0.948 dB.

Visual inspection agrees with the aggregate: makeup and parts of basketball look cleaner, while
horse and guitar are not consistently better. This does not pass the multiscale realizer gate.

## Diagnosis and next control

The token-only architecture inadvertently removed the v1 guide's 3D convolution and global
mean/max context. It preserved within-cell evidence but reduced the guide to independent cell means
plus local attention, so the render objective traded local detail against macro coherence. The
next capacity-matched control keeps the complete v1 cell-RGB spatial/global path and adds token
cross-attention only as a residual detail path. It trains with feature loss alone first, isolating
the architecture before any render-loss fine-tuning.

- `summary.json` and `train_log.jsonl` record the 647.2-second training run.
- `v1_comparison.json` records matched macro and per-class deltas for the selected panel.
- `visual_contract_projection/mark_flow_visual_report.json` contains all per-class metrics.
- The GIFs/contact sheets compare fitted target, carried state, projection control, deterministic
  marks, zero guide, guided raw/projected samples, and shuffled text under the same guide.

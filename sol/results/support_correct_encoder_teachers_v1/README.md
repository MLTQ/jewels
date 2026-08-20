# Support-correct direct-fit teacher subset

Five held-out cooking clips—one per visual style—were independently fitted with the
support-complete tiled renderer for 3,000 steps. Every fit reached the matched 72,000-primitive
budget.

| Macro metric | Result |
|---|---:|
| PSNR | 27.699 dB |
| SSIM | 0.9680 |
| LPIPS | 0.1720 |
| Median anisotropy | 2.431 |
| Median mixed spacetime tilt | 0.397 |

This is a small, deliberately diverse existence proof and a quality ceiling for the encoder audit,
not a population estimate. The direct fits are materially better than the 120-example encoder
(LPIPS 0.1720 versus 0.3916) and use about twice its median mixed tilt. That gap motivates
distillation and adaptive sparsity work without casting doubt on representation capacity.

`qualitative.png` shows source and reconstruction for every style. `report.json`
contains the complete fit history and per-style perceptual/structural metrics. Checkpoints remain on
the GPU host and are omitted from Git.

# Corpus curve v1: a protocol failure worth recording

Intended as the Stage 0 G1 data curve (12/25/50/100/180 training windows, five visual
domains, one frozen 60-clip held-out set). Stopped after two points because the protocol
itself was measuring the wrong thing.

## Result

| Training windows | Final held-out macro PSNR |
|---:|---:|
| 12 | 21.80 |
| 180 | 22.38 |

15x the corpus for +0.58 dB reads as saturation — and that reading is wrong.

## Why the protocol was invalid

Every point trained for a fixed 6,000 steps, so per-window exposure varied 15x: n=12 saw each
window ~500 times, n=180 only ~33. The trajectories show the consequence directly:

| step | n=12 | n=180 |
|---:|---:|---:|
| 1000 | 21.102 | 20.834 |
| 2000 | 20.976 | 21.573 |
| 3000 | 21.755 | 22.013 |
| 4000 | **21.857** (peak) | 22.334 |
| 5000 | 21.817 | 22.343 |
| 6000 | 21.797 (declining) | **22.380** (still rising) |

n=12 peaked at step 4,000 and then overfit; n=180 rose monotonically and was truncated
mid-improvement. The measured gap is therefore a lower bound set by the step budget, not a
property of the data axis.

**Lesson:** frozen step budgets are the right control for model/architecture comparisons (as
used throughout this project) but the wrong one for data curves. Data curves need matched
epochs or train-to-convergence. Recorded so the mistake is not repeated.

## The finding that does survive: per-style spread

n=180 held-out PSNR by visual domain (12 clips each):

| style | PSNR |
|---|---:|
| 3D render | 24.51 |
| claymation | 23.67 |
| photoreal | 22.32 |
| cartoon | 21.71 |
| **anime** | **19.70** |

A 4.8 dB spread, with anime the clear laggard. This independently reproduces the cel-shading
gate's finding: broad flat regions with sharp ink contours are the structurally hard case for
anisotropic Gaussians (flat-region error improves, contour error worsens). Soft-gradient,
volumetric domains (3D render, claymation) are the easy case. This is an allocation/architecture
problem, not a data problem, and connects directly to the fixed-budget fill/contour allocation
work (jewels-v2o). Reporting only the macro mean would have hidden it entirely — style
diversity in the corpus was a deliberate choice and it paid off immediately.

## Next

Replace the fixed-budget curve with (a) one train-to-convergence run at full corpus for the
honest "what does the corpus buy" number, and (b) treat anime as an allocation problem on its
own track.

Artifacts: `n12_summary.json`, `n180_summary.json`, `curve_analysis.png`.

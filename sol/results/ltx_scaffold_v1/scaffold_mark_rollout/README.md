# Autonomous scaffold-to-jewel rollout

This is the first held-out initial-plus-two-continuation rollout in which **every jewel is generated**.
The first window begins from an empty field; windows two and three receive only the immutable field
and stable IDs produced by earlier windows. No fitted jewel feature, fitted background, target birth
cell, or target rank enters inference.

## Selected result

The selected checkpoint is the 2.13M-parameter, 1,024-rank raster-guided mark flow trained for
12,000 steps across 72 UCF views (12 initial and 60 continuation). Sampling uses deterministic
PyTorch 2.12 CUDA kernels, 20 Euler steps, seed 31, the frozen scaffold topology head, and three
16-frame windows. Training took 162.14 seconds on the allocated RTX 2070 SUPER; the RTX 4090 was
not used.

An audit found that the original hard topology projection treated the observed beginning of a clip
as an ordinary birth boundary. It consequently moved every time-cell-zero jewel's finite-support
start to about three temporal standard deviations before frame zero. Even a fully opaque Gaussian
then contributed only about 1.1% at the first frame, causing the blank-to-detail ramp. The selected
run preserves support before frame zero only for time-cell zero of the initial window. All later
cells and continuation windows retain strict projection.

| Four-source macro | Strict initial boundary | Selected censored boundary | Delta |
|---|---:|---:|---:|
| Correct PSNR | 12.408 dB | **14.570 dB** | **+2.162 dB** |
| Correct SSIM | 0.6041 | **0.6306** | **+0.0265** |
| Shuffled PSNR | 8.807 dB | 10.454 dB | +1.647 dB |
| Null PSNR | 9.780 dB | 11.663 dB | +1.883 dB |
| Correct minus shuffled | 3.601 dB | **4.116 dB** | **+0.515 dB** |
| Correct seam / regular change | 1.0351 | **0.9636** | -0.0715 |
| Correct effective jewels / frame | 10,403 | **7,493** | -2,910 |
| Fitted effective jewels / frame | 5,989 | 5,989 | -- |

The lower selected effective count is an improvement: the old opening tails accumulated weak
density without explaining pixels. The generated field now averages 7,493 effective contributors
per frame, near the requested 5k--10k visual regime and much closer to the 5,989 fitted ceiling.

| Source | PSNR before → selected | SSIM before → selected | Selected frame-0 jewels above 5% alpha |
|---|---:|---:|---:|
| Basketball | 14.089 → **16.235** | 0.664 → **0.684** | 8,972 |
| HorseRiding | 11.705 → **15.167** | 0.489 → **0.607** | 9,513 |
| PlayingGuitar | 10.818 → **13.082** | 0.499 → **0.509** | 8,857 |
| ApplyEyeMakeup | 13.020 → **13.796** | 0.765 → 0.723 | 8,586 |

Before the fix, the corresponding generated frame-zero counts were only 106, 127, 244, and 147.
All three windows in every correct/shuffled/null control preserve contiguous stable IDs exactly;
maximum prior-feature and carried-feature error are both 0.0.

## Interpretation

This passes the structural generation gate. Correct LTX scaffolds beat shuffled and null controls
substantially, density stays in the intended range, and continuation boundaries are not visibly or
metrically worse than ordinary motion. The result also proves that the earlier opening washout was
a censored-boundary bug, not evidence that the representation needed still more jewels.

It does **not** pass the final fidelity gate. Correct panels preserve the requested scene/action but
retain moving-subject noise and lose roughly 7.27 dB against fitted jewel ceilings. ApplyEyeMakeup's
PSNR improves while SSIM falls, showing that aggregate density and color can improve while facial
structure becomes less stable. Topology is no longer the dominant error: the next matched ablation
must target foreground, motion-boundary, rare-chroma, and temporal-stability supervision in the
mark realizer while retaining the frozen v1 baseline.

## Artifacts

- `three_window_rollout_contact.png` is the four-source overview.
- Each `*_three_window_rollout.gif` contains scaffold, fitted ceiling, correct, shuffled, and null
  panels over all 48 frames.
- Each matching `*_contact.png` samples the same panel order through time.
- `summary.json` is the complete machine-readable topology, density, seam, render, capacity, and
  stable-state report for the deterministic selected run.
- `strict_initial_summary.json` and the four `strict_initial_*_contact.png` files retain the exact
  seed-31 legacy-boundary control.
- `train_summary.json` records the selected universal mark-flow training run.

Canonical correct generated fields remain on Aine under
`/home/m/jewels/topology/scaffold_mark_flow_v1/deterministic_base_20`; they are omitted here to
avoid duplicating large tensors in Git.

# Lifecycle/appearance factorization

This experiment tests whether moving-subject detail can improve without perturbing the lifecycle,
topology, or identity of the selected autonomous jewel field. It follows the rejected scalar
saliency arm, which gained 0.402 dB and 0.0303 SSIM but raised quiet-region temporal error 3.8%.

## Two-stream control

The selected 12,000-step v1 mark flow is integrated independently from the same Gaussian noise and
owns all topology counts, causal row selection, stable IDs, temporal center, and time-coupled
log-covariance coordinates `(2,5,7,8)`. A second flow supplies a residual only on declared feature
coordinates. Frozen coordinates are copied from the base after every Euler step, topology
projection, and local-to-global covariance transform. Later topology is predicted only from the
base field, so candidate opacity or geometry cannot silently change which jewels exist.

The appearance checkpoint is the already-trained
`feature_spatial_saliency2_stable05_1000/scaffold_mark_flow.pt`. Reusing it makes this a cheap
architectural probe: no new fit can hide whether exact ownership fixes the previous regression.
All comparisons use deterministic PyTorch 2.12 CUDA sampling, 20 Euler steps, seed 31, the same
four held-out LTX classes, three 16-frame windows, and correct/shuffled/null scaffolds on the
allocated RTX 2070 SUPER. The RTX 4090 was not used.

## Dimension and spatial screens

| Candidate residual | PSNR | SSIM | Foreground PSNR | Foreground edge MAE | Motion-boundary MAE | Quiet temporal MAE |
|---|---:|---:|---:|---:|---:|---:|
| Frozen v1 base | 14.5700 | 0.63060 | 10.2252 | 0.68952 | 0.072349 | 0.021452 |
| All spatial/appearance | **14.9713** | **0.65629** | **10.6827** | 0.67613 | **0.072201** | 0.022406 |
| Static spatial/detail | 14.9148 | 0.65379 | 10.6600 | **0.67587** | 0.072205 | 0.022388 |
| Color + spatial gradients | 14.7476 | 0.63766 | 10.4657 | 0.68341 | 0.072240 | 0.021520 |
| Color only, all cells | 14.7550 | 0.63781 | 10.4863 | 0.68263 | 0.072224 | 0.021516 |
| Color only, top 10% salient cells | 14.5896 | 0.63122 | 10.3096 | 0.68667 | 0.072298 | 0.021445 |
| **Color only, top 20% salient cells** | 14.6041 | 0.63140 | 10.3591 | 0.68486 | 0.072268 | **0.021440** |

The unconstrained appearance gain survives exact geometric lifecycle copying, but its quiet error
rises 4.45%. Freezing opacity and RGB time-gradients barely changes that failure. Removing geometry
cuts the regression to 0.32%, showing that spatial changes make fixed-lifetime jewels sweep through
quiet pixels differently. Restricting RGB residuals to the top 20% of cells by scaffold-derived
foreground, motion, chroma, and edge score crosses the predeclared stability gate.

## Selected result

Against its matched frozen base, the selected top-20% RGB residual:

- gains **0.0341 dB PSNR** and **0.00079 SSIM** without changing effective density
  (both remain 7,492.93 contributors/frame);
- raises macro foreground PSNR by **0.1339 dB** and lowers foreground-edge MAE by **0.00465**;
- lowers motion-boundary MAE by **0.000080** and quiet temporal MAE by **0.0000124**;
- improves PSNR in all four classes, and foreground PSNR, foreground-edge MAE, and
  motion-boundary MAE each improve in three classes;
- preserves the correct-versus-shuffled separation (14.604 versus 10.463 dB) and correct-versus-null
  separation (14.604 versus 11.656 dB); and
- is bit-identical to the frozen base in lifecycle coordinates, count tensors, and stable IDs for
  all **12** source/control rollouts.

The macro stability gate passes, though Basketball and ApplyEyeMakeup retain tiny quiet-error
increases of `8.2e-6` and `9.4e-6`; HorseRiding improves by `6.6e-5` and PlayingGuitar by `0.8e-6`.
This is therefore a safe factorization proof, not a claim that moving-subject fidelity is solved.
The gain is intentionally conservative and remains far below the 21.840 dB fitted ceiling.

## Artifacts and provenance

- `color_salient20_summary.json` is the selected complete report.
- `color_salient20_contact.png` compares LTX, fitted, frozen base, selected correct, shuffled, and
  null panels at the first frame of the third window.
- The four `*_three_window_rollout.gif` files retain all 48 frames for the selected arm.
- `all_dimensions_summary.json`, `static_detail_summary.json`, `color_detail_summary.json`,
  `color_summary.json`, and `color_salient10_summary.json` retain every rejected screen.
- Canonical paired fields remain on Aine at
  `/home/m/jewels/topology/scaffold_mark_flow_v1/deterministic_lifecycle_color_salient20_seed31_20`.

## Consequence

Lifecycle factorization is viable, but a full-field appearance residual is too permissive. The
next learned adapter should be zero-initialized over the frozen flow, restricted to appearance
outputs, and trained with scaffold-owned spatial gates plus explicit quiet-region consistency.
That adapter can enlarge the safe residual beyond RGB only; it should not relax the exact
base-owned topology/lifecycle/ID contract established here.

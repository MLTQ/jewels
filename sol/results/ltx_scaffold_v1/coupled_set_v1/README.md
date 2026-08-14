# Neighborhood-coupled jewel birth sets

## Question

Does the autonomous realizer's structured speckle come from processing learned jewel/rank hidden
states independently after the common noisy-mark raster, and can a small shared set state improve
coherence without changing the established topology, carry, density, prompt, or renderer contracts?

## Architecture

The selected v1 mark flow is augmented by one zero-residual set block after rank/noisy-mark and
guide conditioning. The block scatters learned hidden rows into per-cell mean, variance, log-count,
and occupancy, mixes the raster with a 3D neighborhood convolution, then broadcasts one shared
residual back to every addressed jewel. It is permutation-equivariant and linear in jewel count;
there is no padded rank tensor or quadratic point attention.

The block adds **414,016 parameters** to the 2,125,846-parameter base (+19.48%). Every base tensor
is loaded from `run_12000` and frozen. Its final projection initializes to exact zero, so the
augmented model is bit-identical to the selected base before training. Topology prediction,
canonical ranks, stochastic noise, projection, append-only carry, IDs, and rendering are unchanged.

## Training screen

One 3,000-step frozen-base run uses the same 12 UCF training fields, four held-out LTX classes, 72
training views, and 288x192 (3:2) scaffold preparation. It completes in 24.58 seconds on the RTX
4090. Immutable 250-step snapshots prevent late drift from erasing the selected screen.

Step 2,250 has the lowest held-out correct-path feature error: **1.538485** versus **1.541898** for
the exact frozen base on the same seed-47 paths, a 0.221% reduction. Correct-versus-shuffled
separation rises from 0.019254 to 0.019715. This is only a checkpoint-selection diagnostic; rendered
three-window fields remain the decision gate.

## Exact matched 40x24 regression

The old 40x24 protocol is retained only for an exact comparison with the established independent
mark-flow baseline on the RTX 2070 SUPER. Both use deterministic source-stable seeds 31--34, 20
Euler steps, the same frozen topology model, correct/shuffled/null controls, and 48 autonomous
frames per class.

| Mechanism | PSNR | SSIM | Foreground PSNR | Foreground edge MAE | Motion-boundary MAE | Quiet temporal MAE |
|---|---:|---:|---:|---:|---:|---:|
| Independent v1 base | 14.5700 | **0.63060** | 10.2252 | 0.68952 | 0.072349 | 0.021452 |
| Coupled set, full strength | **14.9751** | 0.62616 | **11.1027** | **0.65655** | **0.071758** | **0.020750** |
| Delta | **+0.4051** | -0.00444 | **+0.8774** | **-0.03297** | **-0.000590** | **-0.000702** |

PSNR improves in all four classes. Foreground PSNR improves materially in Basketball,
HorseRiding, and ApplyEyeMakeup and is effectively flat in PlayingGuitar (-0.00038 dB).
Foreground-edge and quiet error improve in all four; motion-boundary error improves in three.
SSIM improves only in Basketball and therefore fails the old-grid structural-similarity criterion.
This is a promising but incomplete screen, not a selected architecture on its own.

## Native-aspect decision

The primary self-consistent 288x192 gate uses the same four source-stable seeds and 20 steps. The
base and candidate use the same frozen topology model, but each topology head sees its own
generated carry after the first window. This is the intended autonomous system comparison, not the
final causal attribution of the set block.

| Mechanism | PSNR | SSIM | Foreground PSNR | Foreground edge MAE | Motion-boundary MAE | Quiet temporal MAE |
|---|---:|---:|---:|---:|---:|---:|
| Independent v1 base | 14.1417 | 0.60009 | 9.9783 | 0.25917 | 0.134010 | 0.023303 |
| Coupled set, autonomous | **14.4753** | **0.60309** | **10.6366** | **0.25167** | **0.133451** | **0.022668** |
| Delta | **+0.3336** | **+0.00300** | **+0.6583** | **-0.00749** | **-0.000560** | **-0.000635** |

PSNR improves in all four classes, SSIM in three, foreground PSNR and edge error in all four,
motion-boundary error in three, and quiet temporal error in all four. The candidate remains in the
intended contributor regime (7,797 effective jewels/frame versus 7,886 for the base). Later count
feedback changes the four-class birth total from 264,870 to 266,730 (+0.70%), so an exact
base-owned topology audit is retained as the causal decision gate. Contact review finds only
subtle improvements: the horse/rider mass and some scene boundaries are more coherent, but all
four autonomous outputs remain far from the fitted ceiling. This is evidence for joint birth
reasoning, not a claim that structured speckle is solved.

A half-strength evaluation-only calibration is also retained: it scales only the set block's
zero-origin final residual projection and records its source and strength in checkpoint metadata.
It is not a retrained or resumable model. The half-strength 40x24 screen was rejected: it reduced
the full-strength gains, still lowered SSIM in three classes, and introduced small motion/quiet
regressions in PlayingGuitar and ApplyEyeMakeup. This rules out simple residual over-strength as
the cause of the old-grid SSIM result.

## Exact-count causal attribution

The paired native audit runs the independent base first and gives its complete per-cell count/rank
sequence to the coupled candidate with a fresh identically seeded noise generator. Every one of
the 115 shared checkpoint tensors must be bit-identical; the 13 new state tensors are the only
learned difference. The candidate still generates its own marks, context, and carry. Results are
therefore attributable to the new mark path rather than a different topology budget.

| Mechanism | PSNR | SSIM | Foreground PSNR | Foreground edge MAE | Motion-boundary MAE | Quiet temporal MAE |
|---|---:|---:|---:|---:|---:|---:|
| Independent v1 base | 14.1417 | **0.60009** | 9.9783 | 0.25917 | 0.134010 | 0.023303 |
| Coupled set, exact counts | **14.4742** | 0.59956 | **10.7390** | **0.25175** | **0.133469** | **0.022582** |
| Delta | **+0.3325** | -0.00052 | **+0.7606** | **-0.00742** | **-0.000541** | **-0.000721** |

All four classes improve PSNR, foreground-edge, motion-boundary, and quiet-temporal error.
Foreground PSNR improves in three, with ApplyEyeMakeup down 0.0346 dB. SSIM improves only in
Basketball and HorseRiding; PlayingGuitar and ApplyEyeMakeup lose 0.00602 and 0.01198. Effective
density stays in the intended regime but falls 2.03% (7,726 versus 7,886 contributors/frame)
because the same rows have changed opacity, covariance, and lifetime marks.

The strict selection gate therefore **fails**: exact-count SSIM improves in only two classes, and
the paired contact does not show a materially more recognizable actor/object in three. The
experiment nevertheless supplies the clearest causal result in this sequence: joint set state
improves every class's pixel fidelity and all three macro error measures without changing a single
count, rank, inherited weight, ID, or carried row. Retain the coupling mechanism as the starting
point for a rendered set/trajectory objective; do not select this feature-loss checkpoint as the
finished realizer.

## Artifacts

- `train_summary.json` and `train_log.jsonl` retain the complete 3,000-step screen and checkpoint
  selection curve.
- `matched_40x24_summary.json` and `matched_40x24_contact.png` retain the exact old-protocol
  correct/shuffled/null regression.
- `strength05_matched_40x24_summary.json` retains the rejected half-strength calibration screen.
- `native_288x192_summary.json`, `native_288x192_contact.png`, and four native GIFs retain the
  primary autonomous gate.
- `paired_native_288x192_summary.json`, `paired_native_288x192_contact.png`, and four paired GIFs
  retain the definitive exact-count causal audit.
- Checkpoints remain on Aine under
  `/home/m/jewels/topology/scaffold_coupled_set_v1/run_3000_seed47_native`.

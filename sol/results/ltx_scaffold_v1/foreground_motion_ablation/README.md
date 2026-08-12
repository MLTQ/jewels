# Foreground and motion supervision ablation

This is a bounded follow-up to the first autonomous three-window scaffold-to-jewel rollout. Every
arm initializes from the selected 12,000-step universal mark flow, fine-tunes for 1,000 updates on
the same 12 UCF training videos, freezes the topology model, and is evaluated on the same four LTX
videos. The decisive baseline/combined comparison uses deterministic PyTorch 2.12 CUDA sampling,
20 Euler steps, and seed 31 on the allocated RTX 2070 SUPER. The RTX 4090 was not used.

## Tested mechanisms

1. **Rendered saliency/stability** oversamples guide regions with foreground deviation, chroma,
   spatial edges, and motion, then penalizes rendered RGB, motion-boundary, and quiet-region change.
2. **All-feature saliency** weights every 22-D jewel velocity by its scaffold-cell saliency.
3. **Spatial/appearance saliency** excludes temporal center and time-coupled covariance dimensions
   `(2,5,7,8)` from the extra weighting while retaining their ordinary feature loss.
4. **Spatial/appearance + stability** combines the third arm with a small rendered quiet-change
   penalty. This is the predeclared final scalar-loss test.

Target-derived saliency masks are used only for loss sampling and evaluation. They do not require
class labels, boxes, segmentation masks, fitted jewel topology, or fitted marks at inference.

## Results

| Deterministic arm | PSNR | SSIM | Temporal-change ratio | Quiet temporal MAE | Foreground PSNR | Motion-boundary MAE |
|---|---:|---:|---:|---:|---:|---:|
| Selected base | 14.570 | 0.6306 | 1.1918 | **0.02145** | 10.225 | 0.07235 |
| Render saliency/stability | 14.600 | 0.6528 | 1.1988 | 0.02325 | 9.254 | 0.07428 |
| All-feature saliency | 14.942 | 0.6380 | 1.2635 | 0.02215 | 10.820 | 0.07241 |
| Spatial/appearance saliency | 14.929 | 0.6416 | 1.2398 | 0.02180 | **10.863** | 0.07210 |
| Spatial/appearance + stability | **14.972** | **0.6610** | **1.1872** | 0.02227 | 10.456 | **0.07201** |

The combined arm is a real appearance improvement: against the exact-seed base it gains 0.402 dB
and 0.0303 SSIM, improves contrast, edge energy, saturation, aggregate temporal-change ratio,
foreground RGB/edge error, and motion-boundary error, and raises PSNR/SSIM in all four classes.
Foreground-edge and motion-boundary error improve in three classes.

It nevertheless fails the predeclared stability gate. Quiet-region temporal MAE rises from 0.02145
to 0.02227 (+3.8%); PlayingGuitar and ApplyEyeMakeup each worsen by about 0.0018. Foreground PSNR
also falls for Basketball and PlayingGuitar, so subject-detail gains are not consistent in three
classes. The branch is rejected and the universal base remains selected.

## Architectural conclusion

Scalar loss weighting has reached a coupling limit. Even when the extra feature loss excludes
temporal lifecycle dimensions, updating the shared flow trunk changes every velocity and the
free-running state feeds those changes into later windows. The next experiment should freeze the
passing base trajectory and learn a separate appearance/geometry residual stream. At sampling
time, temporal center and time-covariance state should be copied from an independently integrated
base stream, making lifecycle behavior an exact control rather than another soft penalty. Only the
appearance stream should receive foreground/chroma/detail supervision.

That two-stream experiment has a simple falsifier: lifecycle dimensions and stable IDs must remain
bit-identical to the base, while foreground detail improves in at least three classes without a
quiet-temporal regression. If it fails, the remaining problem is not loss competition and should
move to a larger/diverse jewelizer corpus or a different conditional representation.

## Artifacts

- `base_summary.json` is the selected deterministic rollout.
- `render_stability_summary.json`, `all_feature_summary.json`, and
  `spatial_feature_summary.json` retain the exploratory arms.
- `spatial_feature_stability_summary.json` contains explicit seed, runtime, checkpoint, manifest,
  and corpus provenance for the decisive exact-seed arm.
- `spatial_feature_stability_contact.png` is its four-source overview.
- `spatial_feature_stability_train_summary.json` records the 94.69-second fine-tune.

# Four-class shared tokenizer: 3,000-step gate

## Protocol

This run repeats the leakage-safe four-class tokenizer experiment from the 500-step smoke with a
fresh learning-rate schedule: 12 source-group 1--3 windows train one shared `32^3` grid, 24-D
rank-conditioned tokenizer, while all four group-4 windows remain held out. Each fitted target has
120,000 jewels. Seventy-five percent of sampled render supervision comes from deterministic
training-source saliency pools split evenly between motion and saturated chroma.

The run completed 3,000 updates in 4,721.8 seconds (78.7 minutes) on the RTX 2070 SUPER. With batch
size one and uniform window sampling, each training window received approximately 250 direct
updates.

## Held-out result

The fixed 4,096-point audit reaches **16.541 dB mean/macro PSNR** and **96.19% count recovery**.

| Held-out source | PSNR | Decoded / target jewels |
|---|---:|---:|
| ApplyEyeMakeup group 4 | 14.853 | 113,837 / 120,000 |
| Basketball group 4 | 16.099 | 116,583 / 120,000 |
| HorseRiding group 4 | 15.445 | 116,439 / 120,000 |
| PlayingGuitar group 4 | 19.765 | 114,856 / 120,000 |

Validation peaked at 16.989 dB in the built-in 512-point audit at step 1,500 and ended at 16.506
dB. The higher-sample final score improves only 0.651 dB over the 500-step smoke and remains below
the 17.786 dB frozen Avenue-to-UCF transfer reference.

The visual gate **fails**. Background layout and broad palette improve, but the action-defining
subjects do not reliably survive: horse/rider and guitarist geometry collapse into texture, and
the basketball player is mostly lost. Prompt training is therefore paused because a prior cannot
restore semantic information removed by its tokenizer.

## Training-source diagnostic

Four source-group-1 windows seen during training reach **19.552 dB** and **98.63% count recovery**.
Their scenes are more recognizable than the held-out set but remain conspicuously noisy. This
3.012 dB train/validation gap shows a generalization problem, while the incomplete training
round-trips also implicate limited per-window exposure or the local set bottleneck itself.

A matched 1,000-update single-window overfit reaches **23.657 dB** in the fixed 4,096-point audit
with **97.60% count recovery**, about 4.1 dB above the shared model on this seen source. Fence
structure and a rough horse/rider silhouette return, proving that exposure matters, but visible
texture noise remains. The result supports an exposure-matched shared schedule while rejecting the
stronger claim that training alone has solved the local codec.

Matched controls remove the 16.78M absolute cell-embedding parameters and compare 24-D versus 64-D
shared Fourier-position cell codes on this same window. The 24-D checkpoint shrinks from 258 MB to
57 MB and reaches **21.822 dB** in the fixed audit; 64-D reaches **21.736 dB** despite using the same
2,097,152 latent numbers as the spatial control below. Their renders are nearly identical and both
are worse than the 23.657 dB learned-position overfit. The absolute table was acting as single-video
content memory, while widening the actual content latent does not cure the rank/moment bottleneck.

The next control redistributes the exact `32^3 × 64` latent-number budget as `64^3 × 8`. It removes
global encoder attention and uses shared Fourier positions, spending representation on spatial
density rather than channels. This is the smallest direct test of occupancy-adaptive tokenization
before implementing a variable sparse cell stream. It completes 1,000 steps in only 365 seconds
and lowers final normalized feature loss from about 0.240 to 0.156. Its built-in render audit reaches
22.134 dB, but count recovery stalls at 87.71% because empty cells dominate the fine-grid count
objective. Occupancy balancing plus a 0.5 count weight resolves that failure: the selected control
completes in 366.7 seconds and reaches **24.586 dB / 97.69% count recovery** in the fixed audit. It
beats the learned-position overfit by 0.929 dB, uses 2.34M model parameters instead of 21.48M, and
keeps the rider more consistently visible across time.

The exposure-matched 12,000-step shared run completes in 73.3 minutes. Its fixed held-out audit
reaches **20.735 dB / 99.794% count** across the four untouched group-4 sources; four seen group-1
sources reach **22.413 dB / 98.405% count**. The small 1.68 dB seen/held-out gap does not explain the
failure: both sets still dissolve action-defining actors into mottled texture.

Compact occupied-group controls then replace one dense moment token per cell with explicit sparse
topology. A four-jewel/32-D control reaches 24.752 dB and exact count; increasing render supervision
raises it to **26.019 dB**. A one-jewel/16-D upper bound reaches only 26.087 dB and remains visibly
noisy. This shows that tiny decoded covariance/opacity errors accumulate across the field even after
group averaging is removed. The project therefore pauses learned jewel reconstruction and trains
the persistent continuation model directly on canonical jewel births. See
`../direct_prompted_streaming_3000/README.md`.

## Artifacts

- `heldout_eval_4096.json`: decisive held-out numerical audit
- `summary.json` and `train_log.jsonl`: complete optimization trajectory
- `manifest.json` and root `*_dense_roundtrip.gif`: four held-out visual comparisons
- `train_shard0/train_shard0_eval_4096.json`: seen-source numerical diagnostic
- `train_shard0/*_dense_roundtrip.gif`: four seen-source comparisons
- `overfit_horse_1000/*`: exposure-control trajectory, audit, provenance, and corrected-label GIF
- `overfit_fourier24_1000/*` and `overfit_fourier64_1000/*`: shared-position width controls
- `overfit_spatial64x8_1000/*`: matched-budget spatial allocation with legacy global count loss
- `overfit_spatial64x8_balanced_1000/*`: occupancy-balanced 0.25 count-weight control
- `overfit_spatial64x8_balanced05_1000/*`: selected 0.5 count-weight control
- `shared_spatial64x8_balanced05_12000/*`: completed shared audit, seen-source diagnostic, and all
  held-out/seen GIFs
- `grouped4x32_horse_1000/*`, `grouped2x24_horse_1000/*`: occupied-group controls
- `grouped4x32_render2_horse_1000/*`, `grouped1x16_render2_horse_1000/*`: render-dominant controls

The archived GIFs were generated immediately before the renderer provenance fix and their target
panels retain the legacy text `45k fitted target`; the manifests and checkpoints record the actual
120,000-jewel targets. `render_dense_tokenizer.py` now derives this label from each example. The
animations will be regenerated with corrected labels after the active overfit control releases the
2070S.

The 258 MB checkpoint remains on the compute host and is intentionally not tracked.

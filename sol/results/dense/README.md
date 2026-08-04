# Dense 45k-jewel tokenizer spike

## Decision

Raising the total budget to 45k was necessary and materially changed the result. A fine `32×32×32`
spacetime raster is the first source-held-out tokenizer to preserve recognizable pedestrians. It is
a tokenizer-mechanics pass, not a final fidelity pass or compact prior-ready representation: 786,432 latent
numbers encode 990,000 fitted-feature numbers (1.26× compression), and 32,768 tokens rule out the
current global-attention prior.

## Protocol

- Date: 2026-08-04
- GPU: allocated RTX 2070 SUPER (`cuda:1`, 8 GB)
- Corpus: 121 Avenue windows, each 64×160×284 with 45,000 fitted jewels
- Leakage control: 86 windows train; all 35 windows from source videos `03` and `05` held out
- Metric: exact all-jewel target/roundtrip renders at fixed continuous `(u,v,t)` samples
- Visual gate: matched 16-frame fitted-target/roundtrip GIFs on both unseen sources

The 45k Avenue fit has roughly 6,158 temporal 3σ support intersections per frame, but the newer
contribution-aware audit finds only 3,922 splats above 5% potential peak alpha and 3,280
opacity-weighted effective contributors. UCF is lower at 3,595 and 2,933. The old support count
therefore overstated useful image density; 45k does **not** yet meet the 5k–10k effective regime.
The direct fitted target is preserved as `dense_target.gif`.

## Selected grid-32 result

- Grid: `32×32×32` (32,768 editable spacetime cells)
- Observed maximum occupancy: 57; configured capacity: 64
- Average occupancy: 1.37 jewels/cell
- Encoder: canonical-rank-conditioned local moments, no quadratic global codec attention
- Decoder: sparse variable-count ranks; evaluates about 45k outputs, not 2,097,152 padded slots
- Model: 8.98M parameters; 24 latent values/cell
- Numeric bottleneck: 990,000 raw values → 786,432 latent values (1.26×)
- Training: 3,000 steps in 219.6 seconds

The full 35-window audit (`grid32_heldout_eval.json`) measured:

| Measurement | Result |
|---|---:|
| Window-weighted mean PSNR | **19.987 dB** |
| Median PSNR | 20.022 dB |
| Macro-by-source PSNR | **19.974 dB** |
| Source `03` mean | 20.017 dB |
| Source `05` mean | 19.931 dB |
| Mean decoded/target jewel ratio | 0.99137 |

`grid32_03_w000000_dense_roundtrip.gif` and
`grid32_05_w000000_dense_roundtrip.gif` are the decisive artifacts. Figures remain blurry and
distorted, but their locations, dark silhouettes, and temporal presence survive on both wholly
unseen camera scenes. Earlier grids erased them completely. This comparison selects a tokenizer at
the current target density; it must not be read as validating that density.

The pass is geometric, not fully appearance-faithful. In an early source-`03` frame, a woman's
bright red coat survives as a correctly located figure but decodes tan. One 24-D cell code must bind
P0 RGB plus nine P1 color-gradient values to every local jewel rank; rare saturated foreground
colors are overwhelmed by 45k uniformly weighted jewel targets and a render loss using only 16
uniform random points per step. This is consistent with regression toward the dominant beige/gray
scene palette, not a channel-order or de-normalization bug. Future tokenizer selection must report
foreground chroma/perceptual error separately from aggregate RGB PSNR.

The checkpoint remains on the training host at
`/home/m/jewels/tokenizer/sol_dense_grid32_rank_3000/autoencoder.pt`.

## What failed, and what it established

| Experiment | Held-out result | Visual result | Conclusion |
|---|---:|---|---|
| 12×12×6 sparse, 128-D | 18.940 dB | People erased | 8.95× bottleneck too coarse |
| 16×16×8 rank-v1 | 19.484 dB | People erased | Rank basis collapsed for ordinary cell counts |
| 16×16×8 rank-v2 | 20.428 dB | People erased | Fixed rank encoding helps metric, not local detail |
| 24×24×16 pooled, 64-D | 19.730 dB | Faint traces only | More locality helps but pooled moments still average |
| 24×24×16 rank-aware | 19.72 dB probes | People erased | Rank binding alone is insufficient at ~5 jewels/cell |
| One-window memorization | 23.14 dB probe | People visible | Confounded by absolute cell embeddings memorizing the volume |
| Fitted-field motion loss | 20.55 dB probes | People erased | Random candidates select fit shimmer/static edges |
| Source-video motion pools | 18.50 dB probes | People erased | Strong foreground loss cannot repair an inadequate cell bottleneck |
| **32×32×32 rank-aware** | **19.987 dB full audit** | **People visible** | Near-one-jewel locality is load-bearing |

The rank-v1 bug normalized ordinary ranks by worst-case capacity, so a typical 22-jewel cell used
only the first 8.6% of a low-frequency basis. Integer-rank Fourier wavelengths 4/16/64 fixed that
error. It improved grid-16 full-audit PSNR by about 0.95 dB, but visual inspection prevented a false
success claim.

## Engineering results

- Sparse decoding removes worst-cell padding while retaining explicit count contracts.
- Canonical packing now uses stable vectorized sorts and prefix offsets; the grid-32 corpus prepares
  immediately instead of looping over hundreds of thousands of Python cell objects.
- Source-video motion sidecars can be generated for all 121 windows in about 29 seconds with bounded
  memory (`motion_pool_manifest.json`). They remain useful for future perceptual supervision even
  though they did not rescue the grid-24 codec.
- The exact renderer chunks 45k covariance eigendecompositions, avoiding an 11.38 GB cuSOLVER
  workspace request on the 8 GB card.
- All 53 spike tests pass in the remote PyTorch environment.

## Validated hierarchy

The proposed next step is now complete. Direct neighbor correlation and mean pooling were too weak,
but a fixed train-only PCA over non-overlapping 2³ blocks retained 94.749% variance with 96 of 192
components. It converts `(32³,24)` to `(16³,96)`: eight times fewer tokens and half as many
numbers. Across the full held-out set it reaches 19.763 dB mean render PSNR, only 0.224 dB below the
fine tokenizer. Both unseen-camera GIFs preserve the same recognizable pedestrian evidence.

See [`../axial/README.md`](../axial/README.md) for the prior and editing experiments built on it.

## Resulting architecture

The implemented model keeps grid-32's local fidelity while avoiding flat 32k-token attention:

1. Grid-32 latents are frozen and cached.
2. A 2³ PCA hierarchy supplies 4,096 coarse tokens.
3. Rotating u/v/t axial attention generates the coarse grid without global 4,096² attention.
4. Dirty coarse blocks are sampled with exact clamping; decoded jewels are merged with protected
   moved constraints.
5. The remaining failures are condition diversity and learned repair quality, not token count or
   editor locality.

Appearance supervision is a parallel remaining issue: moving/salient regions need chroma- or
perceptual-weighted render samples so rare colors do not disappear while geometry survives.

The UCF transfer control in [`../transfer/README.md`](../transfer/README.md) sharpens this result.
Frozen Avenue weights fall to 17.786 dB / 86.06% count on basketball, while the identical
architecture trained on that window reaches 22.316 dB / 97.13%. The bottleneck can represent the
new scene, but absolute-cell/domain specialization prevents frozen transfer; appearance remains
visibly blurry even after same-domain training.

This keeps the user's parallelepiped interaction intact: cursor selections still map to conservative
`(u,v,t)` cells, while hierarchy changes only how those cells are generated and repaired.

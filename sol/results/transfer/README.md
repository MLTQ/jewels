# UCF-101 dense-tokenizer transfer spike

## Decision

The 32³/24-D tokenizer architecture can represent a basketball clip, but the Avenue-trained weights
do not transfer. A diverse text-to-video corpus therefore requires joint tokenizer training (or
careful multi-domain adaptation), not merely running the frozen fixed-camera codec on new classes.

This is a one-window architecture diagnostic, not evidence of UCF generalization.

## Dense target fit

- Source: UCF-101 `Basketball/v_Basketball_g01_c01.avi`, first 64 frames
- Resolution: 64×160×214
- Fitter: 12k initial → 45k final jewels, 8,000 steps, 65,536 voxels/step, kNN 64
- Fitted-video PSNR: **33.97 dB**
- Runtime on allocated RTX 2070 SUPER: **3,420.5 s (57.0 min)**

Two earlier attempts exposed a fixed-chunk `torch.cdist` failure at 45k: first a 2.51 GB contiguous
OOM, then an unspecified launch failure that required a server restart. The first corrected renderer
capped the explicit distance result at 50M pairs. An 8,192×45,000 stress test completed in 0.178 s with
394 MB peak PyTorch allocation; the full fit then completed with roughly 1.36 GB reported GPU use.
The safe cap is stable but about 3.1× slower than the old 18-minute Avenue log, so tiled/fused kNN
and periodic fitter checkpoints were prerequisites for corpus-scale fitting.

A post-restart sustained sweep selected a 100M-pair cap. Through the actual `cull_knn` function,
65,536 queries against 45k centers take 0.390 seconds instead of 0.522 at 50M (identical neighbors),
with an 837 MiB measured peak on the 2070S. Exact atomic fitter recovery has also been implemented
and passes interrupted-versus-uninterrupted equivalence across densification on both CPU and CUDA.

The matched full retime is now complete. The 8,000-step 100M run took **3,406.6 s (56.78 min)**,
reached 45k jewels and **34.014 dB** final sampled PSNR. The prior 50M run took 3,420.5 s
(57.01 min) at 33.970 dB. Thus the isolated 25% culling gain produced only a **13.8 s / 0.4%**
end-to-end improvement. Recovery checkpoints remained healthy throughout and were removed only
after the final checkpoint succeeded. The complete step—not culling alone—must be profiled before
more kernel work. The new fitter also includes exact RNG/recovery mechanics, so this is the relevant
operational comparison rather than a single-variable microbenchmark.

The final field was subsequently rendered at every pixel of all 64 frames against the decoded
source video. Full-volume PSNR is **34.136 dB** (MSE 0.00038581). The direct comparison preserves
court geometry, player motion, and the original orange/teal/red palette; residual error is
concentrated on moving actors, fine lines, and high-frequency ceiling detail. This distinguishes
stage-1 fitting loss from the later tokenizer blur/color regression.

## Frozen Avenue transfer

The native Avenue checkpoint contract failed before inference: UCF reaches 71 jewels in one cell,
above its 64-slot maximum. No jewels were silently dropped. An explicit diagnostic override raised
capacity to 80 while retaining frozen weights, 32³ grid, 24-D latent, and Avenue normalization.

| Measurement | Avenue held-out | Frozen transfer to UCF |
|---|---:|---:|
| Sampled-render PSNR | 19.987 dB | **17.786 dB** |
| Count recovery | 99.14% | **86.06%** |
| Maximum observed cell occupancy | 57 | **71** |

`v_Basketball_g01_c01_w000000_dense_roundtrip.gif` shows a recognizable gym layout but players,
court lines, and saturated colors dissolve into scene-colored haze.

The new-domain feature statistics are not wildly outside Avenue normalization (group RMS z-scores
are 0.94 geometry, 1.05 P0 color, 1.10 P1 gradient). Spatial occupancy differs more strongly:

| Measurement | Avenue mean | UCF window |
|---|---:|---:|
| Occupied 32³ cells | 18,323 | 16,404 |
| Mean p95 occupied-cell count | 8.29 | 11 |
| Mean p99 occupied-cell count | 15.71 | 26 |
| Maximum count | 57 | 71 |

Together with learned absolute cell embeddings, this points to fixed-camera/domain specialization
rather than a simple normalization bug.

## Same-architecture UCF control

The exact 8.98M-parameter architecture was trained from scratch on this one window with an 80-slot
contract. Grid, latent width, compression, and losses otherwise match the selected Avenue run.

| Measurement | Frozen Avenue weights | UCF-trained control |
|---|---:|---:|
| Steps | 3,000 Avenue steps | 3,000 UCF steps |
| 4,096-point sampled-render PSNR | 17.786 dB | **22.316 dB** |
| Best sampled checkpoint metric | — | 22.600 dB at step 2,500 |
| Count recovery | 86.06% | **97.13%** |
| Training time | — | 223.4 s |

The trainer's final 512-point estimate was 22.145 dB; the matched 4,096-point audit above is the
preferred comparison. `ucf_frozen_vs_trained.gif` is the decisive artifact. UCF-specific weights
materially restore court boundaries, wall hoops, player silhouettes, and teal/orange palette. The
result remains glossy and blurred, confirming a second, independent limitation: rare/local
appearance binding still needs stronger perceptual and foreground-chroma supervision.

## Consequences

1. Keep the 32³/24-D hierarchy as a viable starting architecture; do not treat the frozen Avenue
   checkpoint as a universal tokenizer.
2. Raise the corpus contract to at least 80 slots or derive it from a diverse preflight audit.
3. Train tokenizer weights jointly across diverse clips and remove or regularize absolute
   camera-cell memorization.
4. Add salient-motion and chroma/perceptual render sampling so people and rare colors cannot be
   overwhelmed by background pixels.
5. Profile the complete fit step, then fit a class-balanced UCF promptability pilot. At the measured
   rate, 16 windows are approximately 15.1 GPU-hours and 96 windows approximately 90.8 GPU-hours.

## Artifacts

- `ucf_dense_fit_log.jsonl`: dense-fit PSNR, count, and runtime
- `ucf_basketball_transfer_eval.json`: frozen Avenue cross-domain audit
- `v_Basketball_g01_c01_w000000_dense_roundtrip.gif`: frozen transfer
- `ucf_trained_summary.json`, `ucf_trained_log.jsonl`: same-architecture control
- `ucf_trained_eval_4096.json`: matched high-sample control audit
- `ucf_100m_full_retime.json`: matched full-fit timing and quality comparison
- `ucf_100m_full_render.json`: all-pixel/all-frame render metric
- `v_Basketball_g01_c01_w000000_45k_fit_compare_100m.gif`: original pixels, direct 45k fit, and
  5× error heatmap across all 64 frames
- `v_Basketball_g01_c01_w000000_45k_fit_contact_100m.png`: five-frame contact sheet of the same
  direct fit comparison
- `v_Basketball_g01_c01_w000000_ucf_trained_roundtrip.gif`: UCF-trained control
- `ucf_frozen_vs_trained.gif`: three-panel controlled comparison

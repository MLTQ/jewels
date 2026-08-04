# `sol/`: structured jewel video research spike

This folder tests the shortest credible route from the current fitted jewel sets to an editable,
text-conditioned video model. It is deliberately independent of the production-shaped `stprim/`
code: experiments can fail here without destabilizing the existing results.

## Target system

```text
text prompt -> conditional latent flow -> raster-ordered spacetime cells
            -> deterministic jewel decoder -> additive renderer -> video

rendered jewel parallelepiped -> cursor selection -> jewel transform
                              -> dirty source/destination cells
                              -> clamped latent inpainting -> repaired video
```

The parallelepiped is the normalized `(u, v, t)` video volume. A cursor drag selects one jewel or
a group, translates it, and marks both the vacated and destination regions dirty. The eventual
text-conditioned latent model resamples those cells while all untouched cells remain bit-identical.

## Research gates, in order

1. **Trust the renderer.** Euclidean center kNN is not conservative for elongated jewels. The spike
   provides an all-jewel reference renderer and an explicit finite-support renderer whose AABB
   test cannot omit any jewel inside the chosen Mahalanobis radius.
2. **Do not lose dense jewels.** `OccupancyGrid` stores count-aware cell statistics and raises on
   slot overflow. It never silently truncates a 45k-jewel target.
3. **Prove a deterministic bottleneck.** Sparse variable-count decoding and rank-conditioned local
   encoding preserve dense fitted fields without allocating worst-cell padding. Visual inspection
   determines the required raster locality.
4. **Generate from text in latent space.** The low-resolution `RasterFlowPrior` proves conditional
   flow plumbing; the visually successful 32k-cell raster now requires hierarchical or axial flow
   rather than flat global attention.
5. **Make edits local.** `EditPlan` converts a 3D selection and translation into protected moved
   jewels plus a conservative dirty-cell mask.
6. **Make inpainting constrained.** `masked_flow_inpaint` samples dirty latent cells and clamps all
   clean cells after every integration step. Text embeddings fit through the generic condition slot.

## Running the spike

From the repository root:

```bash
python -m unittest discover -s sol/tests -p 'test_*.py'
python -m sol.spike
```

The runnable spike checks 45k-jewel packing, demonstrates the old kNN failure mode, constructs a
cursor-style edit plan, and verifies masked latent clamping. It does not claim learned visual
quality; that requires fitted-corpus checkpoints and GPU training.

## Current result

The current corpus uses 45,000 jewels per 64-frame window. Although about 6.2k temporal 3σ supports
intersect each Avenue frame, contribution-aware auditing finds only 3.9k splats/frame above 5% peak
alpha and 3.3k opacity-weighted effective contributors (UCF: 3.6k and 2.9k). It is therefore still
below the intended 5k–10k effective regime. The selected `32×32×32`, 24-D local tokenizer is the
first source-held-out run to preserve recognizable pedestrians, but it is a shared-codec feasibility
result rather than a visual-fidelity pass. Across all 35 windows from unseen videos `03` and `05`, it reaches
**19.987 dB mean / 19.974 dB macro-by-source sampled-render PSNR** and **99.14% count recovery**.
See [`results/dense/README.md`](results/dense/README.md) for the matched animations, failed coarse-grid
controls, and full audit.

The hierarchy/axial gate now passes computationally. A train-only 2³ PCA block codec reduces the
token grid from 32³ to 16³ (8× fewer tokens, 2× fewer numbers) and retains 94.75% block variance.
Its full held-out render audit reaches **19.763 dB**, only 0.224 dB below the fine tokenizer. The
1.94M-parameter axial flow trains at this resolution on the 2070S and its generated latent energy
distance (**0.2493**) beats CLIP retrieval (0.3164) and scene-mean collapse (0.8473).

Two critical gates still fail. Correct CLIP conditions do not beat shuffled or unconditional ones
(conditional gain -0.00171), and full-generation samples used for local repair wash out the dirty
strip. The end-to-end editor itself works: a real 281-jewel translation dirtied 448 of 4,096 coarse
codes, preserved every clean coarse and fine code with **0.0 max error**, and merged moved jewels as
protected constraints. Explicit mask-aware fine-tuning did not materially improve the repair on the
current 86-window/four-training-camera corpus. See [`results/axial/README.md`](results/axial/README.md)
for metrics and animations.

## Go/no-go measurements for the next phase

- Conservative renderer agrees with the all-jewel reference at the declared truncation tolerance.
- Actual dense-corpus occupancy fits the selected grid/slot budget without overflow.
- Held-out tokenizer renders preserve small moving figures as well as aggregate sampled PSNR.
- A latent prior beats scene-mean and nearest-training-window distribution baselines on held-out
  videos. **Passed for distribution, failed for conditioning.**
- Deterministic carry/commit windows render without a measurable boundary seam; learned
  continuation remains unproven.
- A moved jewel group remains fixed while source and destination neighborhoods are resampled.
  **Mechanical invariant passed; visual repair quality failed.**

The 90k isotropic control is complete: doubling total jewels raises effective UCF contributors only
30% (2.9k → 3.8k) and lowers full-volume PSNR by 1.07 dB. The next representation gate is the matched
90k temporal-preserving spatial-split control. A new persistent-streaming audit shows why raw count
scales so poorly: the 45k fit's mean/median finite-support lifespans are 8.41/5 frames, while the 90k
isotropic fit falls to 5.58/3 and nearly doubles observed births per frame. Stable-ID carry/commit
rendering now matches the monolithic finite-support field within `1.2e-7`; the next data gate is a
single longer joint fit cropped into prefix/future views, before any independently fitted prompt
corpus. More flow steps or small learning-rate sweeps on the same six camera
sources are unlikely to resolve either failed gate. VQ, entropy coding, and LLM-token integration
remain premature. The staged implementation and go/no-go gates are in
[`PROMPTABLE_ROADMAP.md`](PROMPTABLE_ROADMAP.md).

## UCF transfer result

After the server restart, the bounded culling path completed a 45k-jewel UCF basketball fit at
33.97 dB. That conservative 50M-pair run took 57.0 minutes on the 2070S. A subsequent sustained
benchmark selected a 100M-pair cap: through the real culling function it returns identical neighbors
and reduces 65,536-query/45k-center latency from 0.522 to 0.390 seconds at an 837 MiB measured peak.
Periodic fitter recovery is now exact across CPU and CUDA densification boundaries and atomically
saves every 100 steps. A real CUDA CLI kill/restart also produced a bit-identical 220-step result
against an uninterrupted control. The subsequent full 8,000-step retime took **56.78 minutes**
versus 57.01 minutes previously: only a 0.4% end-to-end improvement, despite the isolated 25%
culling gain. The complete differentiable training step now needs profiling. The frozen Avenue
tokenizer still fails cross-domain transfer: its 64-slot contract
was too small for a 71-jewel cell, and an explicit non-dropping 80-slot diagnostic reached only
17.786 dB / 86.06% count recovery.

Training the identical 32³/24-D architecture on the UCF window reached 22.316 dB / 97.13% in 223
seconds and visibly restored court geometry, players, and palette. This says the representation is
viable but the weights are fixed-camera/domain-specialized; the remaining blur and color instability
still require stronger appearance supervision. See
[`results/transfer/README.md`](results/transfer/README.md) and the direct
`ucf_frozen_vs_trained.gif` comparison.

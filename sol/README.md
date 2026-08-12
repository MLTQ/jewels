# `sol/`: structured jewel video research spike

This folder tests the shortest credible route from the current fitted jewel sets to an editable,
text-conditioned video model. It is deliberately independent of the production-shaped `stprim/`
code: experiments can fail here without destabilizing the existing results.

## Target system

```text
text prompt -> pretrained video prior -> low-resolution semantic scaffold
            -> stochastic, render-supervised jewelizer -> persistent jewels
            -> additive renderer -> video

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

**Frontier as of 2026-08-12.** A coherent pretrained-video scaffold is now the selected semantic
bootstrap, and the original `24x40` cross-cell/global raster-guided stochastic mark flow is the
selected jewel realizer. Four matched local-token/render-loss variants failed to dominate it. LTX
generated the complete 16-video prompt scaffold corpus, and all four evaluation scaffolds now have
72k-jewel fits at 28.182--35.158 dB. The UCF-trained realizer passes the leakage-safe LTX transfer
gate at 15.342 dB / 0.6942 SSIM versus 13.777 dB / 0.4160 for deterministic continuation. This
validates video-scaffold-to-jewel transfer. A new UCF-trained topology head then removes fitted
cells/counts/ranks for one LTX continuation stride: correct-scaffold slot F1 is 0.6636 versus
0.6092 shuffled and 0.6091 null, and frozen-flow realization reaches 15.464 dB / 0.6967 SSIM versus
15.492 dB / 0.6997 with oracle topology. It predicts 99.68% of the birth budget, averages 5,822
effective jewels/frame, and preserves carried features exactly. Initial generation and multi-stride
generated-mark rollout remain open. The older results below explain the sequence of gates that led
to this frontier.

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

Two downstream gates still fail. Correct CLIP conditions do not beat shuffled or unconditional ones
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
- Deterministic carry/commit windows render without a measurable boundary seam. **Passed.** A
  prefix-conditioned birth overfit also beats disjoint shuffled/null controls. **Passed as an
  overfit; generalization and free-running rollout remain open.**
- A moved jewel group remains fixed while source and destination neighborhoods are resampled.
  **Mechanical invariant passed; visual repair quality failed.**
- Scaffold-conditioned continuation replaces fitted topology, stays in the 5k--10k effective
  density regime, and approaches the frozen oracle-topology realizer. **Passed for one held-out LTX
  stride; initial state and two-stride generated carry remain open.**

The 90k isotropic control is complete: doubling total jewels raises effective UCF contributors only
30% (2.9k → 3.8k) and lowers full-volume PSNR by 1.07 dB. The next representation gate is the matched
90k temporal-preserving spatial-split control. A new persistent-streaming audit shows why raw count
scales so poorly: the 45k fit's mean/median finite-support lifespans are 8.41/5 frames, while the 90k
isotropic fit falls to 5.58/3 and nearly doubles observed births per frame. Stable-ID carry/commit
rendering now matches the monolithic finite-support field within `1.2e-7`. The subsequent learned
continuation gate also passes on a single 96-frame, 120k-jewel joint fit. The selected cell-local
model uses aligned prefix tokens to reach 19.870 dB sampled future-field PSNR, reduce birth-feature
error by 62.7% versus non-overlapping shuffled prefixes, recover 99.95% of births, and copy carried
jewels with 0.0 error. It beats the matched global-context model by 3.41 dB, showing that spatial
correspondence—not merely a window-level condition—is essential. This is an overfit feasibility
result, not yet a generalizing or free-running generator. See
[`results/streaming_continuation_local/README.md`](results/streaming_continuation_local/README.md).
The four-class prompt preflight is now fixed and passes its text-only gate: 12 train and four
source-group-held-out UCF videos, three training phrasings plus one unseen phrasing per class, and
100% held-out CLIP centroid retrieval with a 0.239 minimum cosine margin. All 16 dense 96-frame,
120k-jewel fits are complete. A shared tokenizer trained for 500 steps with source-derived
motion/chroma samples reaches 15.890 dB and 95.74% count recovery across the four untouched group-4
sources. The full 3,000-step gate improves to 16.541 dB and 96.19% count recovery but remains a
visual failure: broad palette survives while action-defining actors collapse into texture. Known
training sources reach 19.552 dB, exposing both a generalization gap and incomplete reconstruction.
The subsequent matched-budget sweep validates spatial density: a shared-position `64^3 × 8` control
reaches 24.586 dB / 97.69% count, 2.85 dB above `32^3 × 64`, while reducing model parameters from
21.48M to 2.34M. Its completed 12,000-step shared gate reaches 20.735 dB / 99.794% on group-4
holdouts but still erases actors on both seen and unseen sources. Occupied-group and one-jewel-token
upper bounds reach 26.019/26.087 dB with exact topology yet remain visibly noisy, identifying
learned jewel reconstruction as the wrong critical path.

The first direct-jewel prompt model predicts frontier births without a reconstruction codec. On
held-out sources, correct text-only mark MSE is 1.1579 versus 1.1794 shuffled, and free decoded birth
counts change by action prompt, but visible geometry is washed out. A causal audit now locates the
failure: exact target topology improves mean PSNR only 14.060 to 14.232 dB, and predicted geometry
and color independently destroy structure. Oracle-topology stochastic flow restores target edge
energy from 59.8% to 84.4% but produces incoherent texture. Supplying the true future only as a
`24x40` semantic guide reaches **16.555 dB / 90.5% edge energy**, +2.324 dB over deterministic
marks, and restores the held-out macro-layout. The authoritative projection also leaves target
renders unchanged at 100 dB. This rejects the factorized-count head as the next washout fix and
selects a semantic-video-scaffold-to-jewel pathway. Subsequent matched controls reject the tested
local multiscale token path and scalar render-loss tuning: the best weight-0.5 arm raises SSIM and
edge energy but loses 0.307 dB overall and destabilizes PlayingGuitar by 1.574 dB. The original
cross-cell/global raster-guide flow therefore remains the selected transfer model. See
[`results/prompt_smoke/direct_prompted_streaming_3000/README.md`](results/prompt_smoke/direct_prompted_streaming_3000/README.md).
The replacement semantic prior is now operational: official distilled LTX-2.3 generated all 16
balanced train/held-out prompt scaffolds at 768x512x49 with no failures and a 9.40 GiB mean aggregate
GPU peak. Contact-sheet audit preserves recognizable macro-geometry for Basketball, HorseRiding,
PlayingGuitar, and ApplyEyeMakeup. See
[`results/ltx_scaffold_v1/README.md`](results/ltx_scaffold_v1/README.md). Average GPU activity was
low because CPU offload spent most of each 173-second sample rebuilding/streaming weights; actual
two-stage denoising took about 14 seconds. Density-matched 72k fits replay Basketball, HorseRiding,
PlayingGuitar, and ApplyEyeMakeup at 28.182, 31.707, 30.839, and 35.158 dB. Across the four, mean
effective density is 5,966 contributors/frame and the mean 5%-alpha count is 6,883/frame. Guitar's
stricter effective count remains 4,597 despite a strong visual, confirming that density must be a
content-aware diagnostic rather than a blind quota.

On the four unseen LTX scaffolds, guided projected marks beat deterministic continuation by 1.565
dB and 0.2782 SSIM; contrast moves 0.426 to 0.743, edge energy 0.486 to 0.830, temporal-change ratio
reaches 0.981, and birth-cell adherence reaches 99.08%. Horse, guitar, and makeup remain visually
recognizable; Basketball is the blurriest case. Correct and shuffled text are effectively tied once
the true LTX raster is supplied, so prompt semantics still come from the scaffold model.

The next topology gate now also passes for one continuation. The 0.88M head uses scaffold RGB and
carried-state density to predict occupied cells and positive counts. Across all initial and
continuation views, correct-guide slot F1 is 0.6636 versus 0.6092 shuffled, 0.6091 null, and 0.6222
train mean. On the shared 32--48 continuation, it predicts 70,724 versus 70,952 births. Synthesizing
all predicted ranks---rather than discarding ranks without an exact fitted counterpart---recovers
5,822 effective contributors/frame and 15.464 dB / 0.6967 SSIM, within 0.028 dB / 0.0030 of oracle
topology. The visual noise is common to both, so the immediate quality work moves to
foreground/motion-aware mark realization and a generated initial-state/multi-window gate. See
[`results/ltx_scaffold_v1/topology_eval/README.md`](results/ltx_scaffold_v1/topology_eval/README.md).
VQ, entropy coding, and LLM-token integration remain premature. The staged implementation and
go/no-go gates are in [`PROMPTABLE_ROADMAP.md`](PROMPTABLE_ROADMAP.md).

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

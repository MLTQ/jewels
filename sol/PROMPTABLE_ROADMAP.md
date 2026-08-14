# Path to a promptable, editable jewel video model

## Decision

Do **not** attempt to train a general text-to-video foundation model on the 2070S or on the current
12-video training split. The controlled washout audit shows that the next step is also **not** a
better occupancy/count head: exact target topology improves deterministic mark renders by only
0.172 dB. The credible route is a hybrid in which a pretrained text-to-video prior supplies a
coherent, low-resolution semantic scaffold and a stochastic, render-supervised jewelizer realizes
that scaffold as persistent editable state.

The current prior is only *architecturally* prompt-ready. It was trained on mean-pooled **image**
CLIP embeddings, not paired prompt embeddings. CLIP's shared space makes text inference possible as
a diagnostic, but does not remove the image/text modality gap or teach prompt composition. A real
promptable result requires text conditions during training.

```text
prompt ─> pretrained text-to-video prior ─> multiscale low-res video scaffold
                                                  │
past overlap ─> exact carried jewels ─────────────┤
                                                  v
                             stochastic topology + mark jewelizer
                         (cross-attention + render/perceptual loss)
                                                  │
                                      persistent jewel field
                                                  │
                                      additive video renderer

edit = clean field + moved protected jewels + dirty mask + prompt/scaffold
                                      └────────> masked local jewel repair
```

## What is already proven

| Gate | Result | Consequence |
|---|---|---|
| Dense representation | 90k isotropic gives only 3.8k effective UCF contributors and loses 1.07 dB versus 45k | Replace lifespan-shrinking densification before corpus scale-up |
| Local tokenizer | 19.987 dB on held-out Avenue sources | The grid/latent shape is viable on-domain |
| Cross-domain tokenizer | Frozen Avenue weights fall to 17.786 dB; UCF-only control reaches 22.316 dB | Architecture transfers, weights do not; train a shared diverse tokenizer |
| Latent distribution | Axial flow beats retrieval and scene-mean baselines | A small prior can model the 16³ hierarchy |
| Conditioning | Correct CLIP image condition does not beat shuffled/unconditional | More flow steps will not create semantics absent from the corpus |
| Edit mechanics | Clean fine/coarse codes remain bit-identical and moved jewels remain protected | The editor's constraint mechanism is sound |
| Learned repair | Dirty regions wash out | Train on paired masks/edits; full-generation samples are not a repair model |
| Corpus durability | Exact CPU/CUDA recovery across densification, plus a bit-exact real CLI kill/restart; atomic saves every 100 steps | Multi-day fitting is now operationally safe |
| Streaming continuation | Cell-local prefix tokens reach 19.870 dB, reduce mark MSE 62.7% versus disjoint shuffled context, recover 99.95% of births, and preserve carried jewels exactly | Persistent-state plus learned-birth factorization is viable; retain spatial context and test held-out clips/free-running rollout next |
| Washout cause | Exact target topology raises deterministic held-out PSNR only 14.060 to 14.232 dB; predicted geometry and color each destroy structure | Do not spend the next experiment on count heads; replace paired deterministic mark regression |
| Stochastic marks | Oracle-topology flow restores edge energy from 59.8% to 84.4% but produces incoherent texture | Sampling avoids some conditional averaging, but still needs a global semantic scaffold |
| Semantic scaffold | A true-future `24x40` guide raises held-out jewel renders to 16.555 dB and 90.5% target edge energy | Build a pretrained-video-guided jewelizer before autonomous topology or direct native generation |

## Phase 1 — representation and codec diagnosis (completed)

This phase established useful locality and failure modes, but learned jewel reconstruction is no
longer the generation critical path. Preserve these gates as regression tests rather than returning
to codec sweeps.

Start with UCF-101 because its directory class is a free, unambiguous action prompt and the data is
already present. Use source groups—not adjacent clips—as the split unit.

1. Smoke corpus: 4 visually distinct classes × 4 source groups = 16 windows.
2. Promptability pilot: 8–12 classes × 8–12 source groups = 96–144 windows.
3. Validate the temporal-preserving 90k spatial-split control; then derive corpus jewel and slot contracts
   from measured effective contributors and occupancy rather than raw total count.
4. Train one tokenizer across all classes. Regularize or replace absolute cell embeddings so it
   cannot solve reconstruction by memorizing fixed cameras.
5. Mix uniform render samples with motion/saliency and saturated-chroma samples. The red-coat
   failure shows that uniform RGB MSE lets rare appearance lose to background statistics.

**Gate:** held-out source groups must preserve recognizable actors and class-specific color/motion;
count recovery should remain above 95%, and performance must materially beat the frozen Avenue
17.786 dB transfer result. Report macro-by-class metrics so large/easy classes cannot hide failures.

The matched 45k full retime is 56.78 minutes/window, but that density is no longer the planned corpus
contract. Re-estimate the 16- and 96-window costs from the 90k control before requesting scale
compute. Atomic 100-step recovery remains mandatory for every long fit.

## Phase 2 — make video-to-jewel realization coherent

The deterministic direct-birth model, exact-topology audit, stochastic mark flow, and oracle guide
now form a causal chain:

1. target topology alone does not remove washout;
2. stochastic marks recover edge energy but not coherent objects;
3. a low-resolution true-future guide restores recognizable global layout.

The next bounded implementation should therefore improve the jewelizer while keeping future video
and target topology privileged. Replace the cell-mean guide convolution with multiscale spatial
features and cross-attention from each cell/rank query. Train the stochastic 22-D mark flow with a
sampled differentiable render loss on its denoised target estimate, plus edge/chroma/perceptual
sampling. Keep feature losses for covariance stability, but do not let them dominate rendered
appearance.

**Go gate:** on all four held-out group-4 videos, guided jewels must visibly preserve action-defining
geometry, improve LPIPS or another perceptual metric as well as PSNR, and substantially narrow the
gap to fitted targets without increasing temporal flicker. Correct topology must remain exact and
the carried prefix must remain bit-identical.

This phase is an amortized video-to-jewel representation model, not text-to-video. That separation is
intentional: it proves that coherent raster semantics can be converted into editable jewels before
we entangle the problem with prompt understanding.

## Phase 3 — replace the oracle guide with a prompt-generated scaffold

Once the video-to-jewel gate passes:

1. Feed the jewelizer low-resolution videos sampled from a frozen pretrained text-to-video model.
   This immediately provides open-vocabulary scene/action knowledge without pretending that 12 UCF
   training clips can teach it.
2. Train on both real-video scaffolds and teacher-generated scaffolds so the jewelizer tolerates the
   teacher's artifacts and distribution while retaining a measurable real-video ceiling.
3. Learn occupied-cell probability and positive count **conditioned on the scaffold**, local overlap,
   and target contributor-rate feedback. Count is now solved in a semantically anchored coordinate
   system rather than expected to create semantics itself.
4. Use overlapping teacher windows and copy carried jewels exactly. Condition the next scaffold on
   the previous raster overlap, then let the jewelizer emit only frontier births.
5. Evaluate correct/shuffled/null text at the scaffold and final jewel render separately. This
   distinguishes teacher prompt control from jewelizer fidelity.

After that works, build a 1k--10k captioned jewel corpus and distill the teacher/scaffold pathway into
a direct text-to-jewel hierarchy. Store caption text and encoder identity with every fit; replace a
single pooled CLIP vector with token-sequence cross-attention for object/action/color binding.

Modern systems reinforce these choices: CogVideoX uses a spatiotemporal VAE, deep text/video
fusion, progressive training, and an extensively filtered/recaptioned corpus; HunyuanVideo likewise
treats data curation, text encoding, architecture, and scaling as one system rather than a denoiser
alone. Their data scale also makes clear why this project should inherit knowledge instead of trying
to recreate it on one GPU.

At thousands of windows, per-video optimization becomes the bottleneck. Before that scale, replace
pure-PyTorch all-center culling with a tiled/fused CUDA neighbor search or rasterizer. Then distill
the fitted targets into a feed-forward video-to-jewel encoder so new training videos do not each
require thousands of optimization steps.

## Phase 4 — train the operation the editor actually needs

Full generation and local repair are different conditional distributions. Build training tuples of:

```text
(clean context codes, dirty source/destination mask, protected moved jewels,
 edit/prompt text) -> target repaired codes
```

Train with random cuboids first, then synthetic translations with known targets. The model must see
the mask and protected-jewel summary; the sampler continues clamping clean codes after every flow
step. Counterfactual move targets can initially come from controllable synthetic scenes or a video
editing teacher, since ordinary footage does not contain paired “same scene, object moved” truth.

**Gate:** zero error outside the dirty region, zero displacement of protected jewels, improved dirty
region render quality over full-generation and nearest-neighbor baselines, and no temporal seam at
the repair boundary. RePaint and Dreamix support the underlying strategy: retain known
spatiotemporal information while a generative prior synthesizes only what must change.

## Phase 5 — interactive parallelepiped

The viewer can be built once Phase 2 emits recognizable prompted samples, but it is not on the
research critical path. It needs:

- WebGPU rendering of the `(u,v,t)` volume and stable jewel IDs;
- ray/cuboid/lasso selection with group transforms;
- conservative dirty masks covering vacated and destination support;
- a server call carrying prompt, clean codes, protected jewels, and dirty mask;
- before/after video and volume views with undoable edit operations.

## Immediate execution order

1. ~~Retime one full 45k fit and exercise real process restart.~~ Both pass; runtime is 56.78 minutes.
2. ~~Complete the isotropic 90k UCF density control.~~ It remains under-dense and loses PSNR; run
   the temporal-preserving spatial-split control instead of scaling raw count again.
3. ~~Define and test the persistent carry/commit contract.~~ Stable global IDs now partition each
   stride into carried jewels and births; finite-support streamed rendering matches the monolithic
   field within `1.2e-7`. The 90k isotropic audit exposes a 3-frame median lifespan and 1,290
   observed births/frame.
4. ~~Jointly fit one 96–128-frame clip and train a continuation overfit that predicts births rather
   than whole windows.~~ The 96-frame/120k field and four 32-prefix/16-future views pass the
   correct/disjoint-shuffled/null gate. The selected cell-local model reaches 19.870 dB, 99.95%
   independently decoded birth density, and 0.0 stable-ID carry error.
5. ~~Add class-balanced UCF enumeration and text-condition sidecars.~~ The fixed 16-video manifest
   holds out group 4 in every class; unseen prompt templates retrieve the correct CLIP class
   centroid at 100% accuracy with a 0.239 minimum margin. All 16 96-frame/120k-jewel fits are now
   complete, and the `32^3` tokenizer audit finds zero slot overflow.
6. ~~Train the shared tokenizer with motion/chroma sampling and compare foreground color against the
   current UCF control. The bounded smoke passes trainability, but the 3,000-step run reaches only
   16.541 dB / 96.19% count on the four held-out sources and still erases action-defining subjects.
   A matched single-window sweep shows that spatial density, not more channels, is decisive:
   `64^3 × 8` reaches 24.586 dB / 97.69% count while `32^3 × 64` reaches 21.736 dB at the identical
   latent-number budget.~~ The exposure-matched shared run reaches 20.735 dB / 99.794% on group-4
   holdouts but fails visually on seen and unseen sources. Sparse grouped controls reach 26.087 dB
   and exact count yet remain noisy, so learned jewel reconstruction is removed from the critical
   path.
7. ~~Train direct prompt-conditioned jewel births while copying carried state exactly.~~ The first
   0.84M-parameter model establishes held-out prompt-to-density and small prompt-to-mark selectivity:
   correct text-only mark MSE is 1.1579 versus 1.1794 shuffled, and free counts change
   deterministically by action prompt. Visual action geometry remains washed out. The exact-topology
   decomposition rejects occupancy/count factorization as the washout fix: oracle topology adds only
   0.172 dB.
8. ~~Test stochastic mark generation and a semantic-scaffold control.~~ Oracle-topology mark flow
   restores edge energy but not coherence; a true-future `24x40` guide reaches 16.555 dB and 90.5%
   target edge energy, +2.324 dB over deterministic marks. This selects the cross-cell/global raster
   guide for matched local-token and render-objective controls before replacing its oracle source
   with a pretrained text-to-video scaffold.
9. ~~Install and validate the official distilled LTX-2.3 pipeline on Aine as the first
   prompt-generated scaffold.~~
   CUDA 13.2 and the RTX 4090 pass preflight; the pinned environment, 46.15 GB distilled v1.1
   checkpoint, 1.0 GB upscaler, and five-shard 24.38 GB Gemma encoder are installed. The balanced
   12-train/4-held-out scaffold gate completed 16/16 videos with no failures at `512x768`, 49 frames,
   FP8 cast, and CPU offload. It averaged 172.98 seconds and 9,399 MiB peak GPU memory per clip; all
   four class audits preserve the requested action and coherent macro-geometry.
10. ~~Fit the four LTX evaluation scaffolds at density matched by spacetime volume.~~ The frozen
    49-frame/72k contract reaches 28.182, 31.707, 30.839, and 35.158 dB for Basketball,
    HorseRiding, PlayingGuitar, and ApplyEyeMakeup. Mean effective density is 5,966/frame and mean
    5%-alpha count is 6,883/frame. Guitar's 4,597 effective contributors/frame despite a strong
    visual confirms that measured density is a content-aware diagnostic, not a blind quota.
11. ~~Test multiscale/render controls and the selected v1 realizer on unseen LTX scaffolds.~~ The
    first token-only multiscale plus render-loss jewelizer is a negative control: versus the v1
    cell-RGB guide it raises contrast 0.856 to 0.869 and edge energy 0.905 to 0.911, but lowers PSNR
    16.555 to 16.419, SSIM 0.8475 to 0.8441, saturation, and temporal stability. It had removed the
    v1 guide's cross-cell 3D/global path. The corrected feature-only hybrid retains that path but is
    also rejected: it reaches 16.240 dB / 0.8345 SSIM and lowers every aggregate appearance metric.
    Raster-only render-loss controls at weights 2.0 and 0.5 also fail the joint gate. Weight 0.5
    raises SSIM 0.8475 to 0.8557 and edge ratio 0.9050 to 0.9251, but loses 0.307 dB overall and
    destabilizes PlayingGuitar by -1.574 dB with excess motion. The retained v1 model then passes
    the predeclared UCF-train/LTX-validation gate: 15.342 dB / 0.6942 SSIM versus 13.777 dB /
    0.4160 deterministic, with contrast 0.743, edge ratio 0.830, 99.08% birth-cell adherence, and
    three of four recognizable actions. Correct and shuffled text remain tied under the true LTX
    guide, so the result licenses scaffold-to-jewel realization only.
12. ~~Learn scaffold-conditioned occupied cells, counts, and birth ranks while carrying stable-ID
    overlap jewels exactly.~~ A 0.88M UCF-trained head reaches 0.6636 slot F1 on all held-out LTX
    views versus 0.6092 shuffled, 0.6091 null, and 0.6222 train mean. On the 32--48 continuation it
    predicts 70,724 versus 70,952 births. The frozen realizer synthesizes every learned rank at
    15.464 dB / 0.6967 SSIM, within 0.028 dB / 0.0030 of oracle topology, while averaging 5,822
    effective contributors/frame and preserving carried features at 0.0 error. Exact fitted-rank
    retention had falsely suggested a density shortfall because Gaussian decompositions are
    non-unique.
13. ~~Generate the initial jewel state and run at least two learned-topology/generated-mark strides
    with only model-produced carry.~~ One 1,024-rank flow now generates an empty-state initial
    stride plus two continuations from append-only model state. The opening washout was a censored
    boundary bug: permitting pre-frame-zero support only in the initial time-cell-zero raises
    visible frame-zero density from 106--244 to 8,586--9,513 without retraining. Correct held-out
    LTX rollouts reach 14.570 dB / 0.6306 SSIM, +2.162 dB over an exact deterministic
    strict-boundary control and +4.116 dB over shuffled scaffolds. Effective density is 7,493/frame,
    every stable ID/carry audit is exact, and seam change is 0.964 times ordinary change.
14. ~~Test scalar foreground, motion-boundary, rare-chroma, and temporal-stability supervision
    while preserving the v1 checkpoint.~~ Four 1,000-step initialized arms establish a genuine but
    coupled detail/stability tradeoff. The best combined arm reaches 14.972 dB / 0.6610 SSIM,
    improves all four classes in both global metrics, and slightly improves aggregate
    motion-boundary/excess-motion error, but raises quiet-region temporal MAE 3.8% and worsens
    foreground PSNR in two classes. Reject it under the predeclared gate and retain v1.
15. ~~Factor lifecycle from appearance during sampling.~~ An independently integrated v1 stream now
    owns topology, stable IDs, temporal center, and time-coupled covariance exactly. Full-field
    residuals retain the earlier +0.401 dB gain but still raise quiet error 4.45%, localizing the
    problem to appearance feedback rather than lifecycle drift. Restricting the residual to RGB in
    the top 20% scaffold-salient cells passes the deterministic four-class gate: +0.034 dB PSNR,
    +0.0008 SSIM, +0.134 dB foreground PSNR, lower macro foreground-edge/motion/quiet errors,
    detail gains in three classes, and bit-identical lifecycle/counts/IDs across 12 controls.
16. Train a zero-initialized appearance adapter over the frozen flow rather than storing a second
    full model. Preserve the exact two-stream ownership and scaffold gate, then expand beyond RGB
    only when the four-class quiet-stability gate remains passing. Evaluate longer rollouts and a
    larger held-out scaffold corpus before promoting it beyond a factorization proof.
17. Measure the fitted-data scaling curve: class-balanced 4/8/12-source realizer+topology stacks
    at the frozen recipe, deterministic native rollouts, velocity/PSNR/SSIM/LPIPS per point. The
    curve's shape decides whether the next compute goes to data or to architecture.
18. Fit the twelve LTX training clips at the frozen 72k contract and retrain the realizer
    domain-matched, comparing against the UCF-trained transfer under the identical battery. This
    moves the data axis onto teacher-generated corpus, which is the axis donated compute extends.
19. Distill the teacher's guide into a compact text-to-scaffold generator trained on harvested
    (prompt, scaffold-raster) pairs — the smallest new model that makes inference prompt-only —
    and evaluate correct/shuffled text at the scaffold and at the final render separately.

## Compute conversion — the scaling program (2026-08-14)

Reframe (Max): editability is a supporting property, not the headline. The goal is a promptable
text-to-video model whose generation is the composition of anisotropic spacetime Gaussians —
primitives with learned shape in every dimension, motion carried by orientation — and the
scaffold pathway is how that model trains without pretending one workstation can learn
open-vocabulary semantics from scratch. What a compute donor needs to see is that additional
training converts into quality at a measured rate. Three facts make that conversion mechanical:

1. **Teacher supply is unlimited.** LTX-2.3 emits a prompt-matched 49-frame video in about
   three minutes on the 4090, with seeds, receipts, and balanced prompt manifests already
   automated (`generate_ltx_corpus.py`).
2. **Supervision conversion is ~25 GPU-minutes per clip.** The frozen 72k contract turns any
   such video into a (scaffold, jewel-field) training pair at 28--35 dB replay, with resumable
   fitting. One 4090 day ≈ 50 new supervised fields including generation; the current realizer
   corpus is twelve.
3. **The realization stage is data-starved by construction.** The 2.13M mark flow trains in
   under three minutes and the 0.88M topology head in under one; both are dwarfed by their own
   corpus cost. Nothing about the current quality ceiling has ever seen a data axis.

The running experiments make the conversion measurable rather than asserted:

- **Fitted-data scaling curve.** Class-balanced 4-, 8-, and 12-source realizer/topology stacks at
  the exact frozen recipe/seed/battery, each rolled out deterministically at native 288x192 and
  scored with velocity loss, PSNR/SSIM, and LPIPS (`topology/scaling_curve_v1`,
  `sol/results/ltx_scaffold_v1/data_scaling_v1`). A monotone, unsaturated curve is the pitch; a
  saturating one redirects the next dollar to architecture instead — either answer is useful.
- **Domain-matched LTX corpus.** The twelve LTX training clips are being fitted at the frozen 72k
  contract (`corpus/ltx_scaffold_v1_train_72k`), so the realizer can next train in the same
  domain it is evaluated in, removing the measured UCF-to-LTX transfer penalty and putting the
  curve on the axis that donated compute actually extends: teacher-generated data.

With donated compute the program is, in order: (a) generate-and-fit at fleet scale — at the
current fitter, eight datacenter GPUs convert to roughly three thousand supervised fields per
week, before the already-planned fused-CUDA culling or amortized video-to-jewel encoder removes
the 25-minute bottleneck; (b) re-train the realizer per corpus doubling under the same frozen
battery and publish the extended curve; (c) distill the teacher out of inference — the entire
inference-time teacher signal is a `(16,16,8)` cell-RGB raster per 16-frame stride, about six
kilobytes, so a compact text-conditioned scaffold generator trained on harvested (prompt,
scaffold) pairs makes the system prompt-only at inference while the jewel stack stays unchanged;
(d) scale realizer capacity with the corpus, which is exactly where the coupled birth-set result
already points.

## Literature anchors

- [CogVideoX: Text-to-Video Diffusion Models with an Expert Transformer](https://arxiv.org/abs/2408.06072)
- [HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603)
- [Goku: Flow Based Video Generative Foundation Models](https://arxiv.org/abs/2502.04896)
- [RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865)
- [Dreamix: Video Diffusion Models are General Video Editors](https://arxiv.org/abs/2302.01329)
- [Phenaki: Variable Length Video Generation From Open Domain Textual Description](https://arxiv.org/abs/2210.02399)
- [StreamingT2V](https://openaccess.thecvf.com/content/CVPR2025/html/Henschel_StreamingT2V_Consistent_Dynamic_and_Extendable_Long_Video_Generation_from_Text_CVPR_2025_paper.html)
- [AdapTok](https://openaccess.thecvf.com/content/CVPR2026/html/Li_AdapTok_Learning_Adaptive_and_Temporally_Causal_Video_Tokenization_in_a_CVPR_2026_paper.html)
- [Unlocking Point Processes through Point Set Diffusion](https://arxiv.org/abs/2410.22493)
- [DiffSplat: Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation](https://arxiv.org/abs/2501.16764)
- [Not-So-Optimal Transport Flows for 3D Point Cloud Generation](https://arxiv.org/abs/2502.12456)

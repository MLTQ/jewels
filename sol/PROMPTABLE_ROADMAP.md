# Path to a promptable, editable jewel video model

## Decision

Do **not** attempt to train a general text-to-video foundation model on the 2070S. The credible
route is to prove that the jewel representation can serve as an editable latent space underneath a
small conditional generator, then inherit broad language/video knowledge from frozen encoders,
captioned data, and eventually a pretrained video teacher.

The current prior is only *architecturally* prompt-ready. It was trained on mean-pooled **image**
CLIP embeddings, not paired prompt embeddings. CLIP's shared space makes text inference possible as
a diagnostic, but does not remove the image/text modality gap or teach prompt composition. A real
promptable result requires text conditions during training.

```text
captioned video
      │
      ├─ high-quality fit ─> 45k-jewel target ─> shared tokenizer ─> 16³ coarse + fine codes
      │                                                               │
prompt ─> frozen text encoder ─────────────────────────────────────────┤
                                                                      v
                                                        conditional latent flow
                                                                      │
                                                   coarse-to-fine jewel decoder
                                                                      │
                                                      additive video renderer

edit = clean codes + moved protected jewels + dirty mask + prompt
                                      └────────> masked conditional flow repair
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

## Phase 1 — a diverse, shared jewel tokenizer

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

## Phase 2 — prove prompt selectivity before open vocabulary

Train the existing 16³ axial rectified-flow prior on **text embeddings of the UCF labels and prompt
templates**, not image embeddings. Keep classifier-free condition dropout. For this small closed
vocabulary, the current pooled CLIP-vector condition is sufficient and cheap.

Use multiple templates per class (for example, “a person playing basketball” and “an indoor
basketball game”) so the model cannot key on one fixed string. Hold out source videos and some
templates, but not whole classes at first.

**Required controls:**

- correct prompt versus shuffled prompt versus null prompt on the same flow paths;
- generated-video action classification/retrieval versus chance and nearest-training retrieval;
- within-prompt diversity, to reject memorized class prototypes;
- macro-by-class rendered inspection, especially small actors and saturated clothing.

**Go gate:** correct text must beat both shuffled and unconditional conditioning by a stable margin,
and generated class accuracy must beat both chance and the unconditional model. If this fails, do
not scale the model; inspect text/latent mutual information and class balance first.

This phase is a promptable *domain model*, not a general text-to-video model. That limited claim is
valuable: it isolates whether semantic control survives the jewel tokenizer.

## Phase 3 — move from class prompts to natural captions

Once the class gate passes:

1. Build a 1k–10k single-shot captioned corpus from a curated public subset or videos generated by
   a pretrained text-to-video teacher. Teacher data proves representation and control; it does not
   make the student more knowledgeable than the teacher.
2. Store the caption text and encoder identity with every fit. Avoid anonymous `.npy` conditions.
3. Replace the single pooled CLIP vector with token-sequence conditioning from a frozen language
   encoder and cross-attention in the axial blocks. Pooled vectors are adequate for class labels but
   poor for binding multiple objects, actions, colors, and camera instructions.
4. Freeze the now-diverse tokenizer while training the prior. Jointly tune only after prompt
   selectivity is established, or codec drift will confound the generator experiment.
5. Add coarse-to-fine generation: sample the 16³ global code first, then predict local fine residuals
   conditioned on coarse code and text. This preserves the validated hierarchy and avoids a flat
   32³ global attention problem.

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
6. Train the shared tokenizer with motion/chroma sampling and compare foreground color against the
   current UCF control. The bounded smoke passes trainability, but the 3,000-step run reaches only
   16.541 dB / 96.19% count on the four held-out sources and still erases action-defining subjects.
   A matched single-window sweep shows that spatial density, not more channels, is decisive:
   `64^3 × 8` reaches 24.586 dB / 97.69% count while `32^3 × 64` reaches 21.736 dB at the identical
   latent-number budget. The exposure-matched 12,000-step shared spatial gate is running; convert
   the winner to occupied fine tokens under a coarse hierarchy only after held-out renders pass.
7. Expand only after the smoke gate, then run the correct/shuffled/null prompt experiment.

## Literature anchors

- [CogVideoX: Text-to-Video Diffusion Models with an Expert Transformer](https://arxiv.org/abs/2408.06072)
- [HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603)
- [Goku: Flow Based Video Generative Foundation Models](https://arxiv.org/abs/2502.04896)
- [RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865)
- [Dreamix: Video Diffusion Models are General Video Editors](https://arxiv.org/abs/2302.01329)
- [Phenaki: Variable Length Video Generation From Open Domain Textual Description](https://arxiv.org/abs/2210.02399)
- [StreamingT2V](https://openaccess.thecvf.com/content/CVPR2025/html/Henschel_StreamingT2V_Consistent_Dynamic_and_Extendable_Long_Video_Generation_from_Text_CVPR_2025_paper.html)
- [AdapTok](https://openaccess.thecvf.com/content/CVPR2026/html/Li_AdapTok_Learning_Adaptive_and_Temporally_Causal_Video_Tokenization_in_a_CVPR_2026_paper.html)

# Time-distorted Gaussian-splat text-to-video feasibility report

## Executive decision

**Conditional go.** The project now has decision-grade evidence that its distinctive representation
works, can be optimized at useful scale on one RTX 4090, and can be produced by a shared encoder
whose held-out quality continues to improve with more balanced training data. Prompt identity also
survives the pretrained-video-to-splat bottleneck.

This is enough to pitch a credible compute-scaling program. It is not yet evidence that the system
can generate a video directly from text: the largest remaining scientific gate is a
prompt-conditioned model that predicts the encoder's splat latent without receiving target-video
pixels.

## What is now established

| Claim | Evidence | Decision |
|---|---|---|
| Time-distorted geometry is useful | Free spacetime tilt beats axis-aligned geometry in 9/9 matched real-video fits by +0.772 dB mean, 95% CI [+0.573, +0.971] | Passed |
| Correct rendering is computationally usable | Support-complete tiled rendering stays within 2× KNN fit-step time at 10k, 45k, and 72k primitives | Passed |
| The shared video-to-splat encoder scales | Held-out PSNR rises 21.876 → 22.399 → 22.628 dB at 12/60/120 balanced examples, across three seeds | Passed |
| Perceptual and structural quality scale | From 60 to 120 examples: LPIPS falls 0.4651 → 0.3916, PSNR and SSIM rise, and median mixed spacetime tilt rises 0.0725 → 0.1875 | Passed |
| Prompt identity survives splat encoding | On 12 held-out prompts, correct text beats shuffled text in 12/12 renders; mean cosine margin +0.1913; 91.4% of source-video alignment is retained | Passed for the scaffold path |
| Text alone can generate the latent | No direct prompt-to-latent generator has been trained in this corrected representation | Not yet tested |

## Encoder scaling experiment

The primary curve uses exact nested subsets of 12, 60, and 120 examples, three training seeds per
budget, and a frozen 60-video validation set. At 60 examples there is exactly one example in every
style/action stratum; at 120 there are exactly two. The 12-example arm covers every action and
rotates across all five styles. Every arm trains for equal corpus passes rather than an equal raw
step count, then receives the same low-rate convergence continuation.

| Training examples | Held-out PSNR mean | 95% CI | LPIPS | Layout PSNR | Median mixed tilt |
|---:|---:|---:|---:|---:|---:|
| 12 | 21.876 dB | [21.751, 22.000] | 0.4728 | 29.812 dB | 0.0479 |
| 60 | 22.399 dB | [22.344, 22.453] | 0.4651 | 30.726 dB | 0.0725 |
| 120 | 22.628 dB | [22.459, 22.797] | 0.3916 | 31.533 dB | 0.1875 |

Every 120-example seed beats every 60-example seed. The two 95% intervals narrowly do not overlap.
The largest point improves every measured perceptual/layout metric and increases rather than
collapses the representation's defining mixed space/time geometry.

The direct-fit teacher subset remains substantially better (27.699 dB, LPIPS 0.1720, median mixed
tilt 0.3966). This is useful headroom: it shows that the representation can express more quality
than the encoder currently extracts.

## Why prior negative documentation was not treated as law

Two protocol defects were found in the previous encoder evidence:

1. The old small subset was produced by lexicographic truncation. Its 12 examples were all anime,
   despite being described as broadly representative.
2. A fixed 6,000-step comparison exposed each 12-example clip about 500 times but each 180-example
   clip only about 33 times. The small arm could be declining while the large arm was still
   improving, so the curve mixed data scale with unequal convergence.

The new experiment corrects both defects and uses replicated seeds. An early support-capacity
overflow also stopped loudly and was rerun from scratch at a larger bound; no candidate truncation
was accepted as data. These findings invalidate the old experiment's general conclusion, not every
individual observation it recorded.

## Remaining gaps and risks

- The prompt smoke test uses the lowest-risk route: prompt → pretrained video scaffold → shared
  encoder → support renderer. It proves semantic retention, not direct conditional generation.
- All 73,728 encoder slots are active above the current opacity threshold. Sparse count prediction
  and pruning are not yet working, so representation size is fixed rather than content-adaptive.
- The encoder-to-teacher gap is still large: LPIPS 0.3916 versus 0.1720 on the five-style audit.
- The 120-example curve is encouraging but too short for a trustworthy asymptotic compute law or
  a production-quality projection.
- Prompt controls cover 12 actions in the photoreal validation style. Broader style/prompt
  composition and human preference evaluation remain unmeasured.

## Recommended next gate

Freeze the 120-example encoder and train a small conditional flow or diffusion model over its
structured output: lattice residuals plus the seed-RGB grid. Use the existing 60 style/action
strata and explicit correct, shuffled, and dropped-text controls. Evaluate generated support
renders—not just latent losses—on held-out source groups.

The predeclared pass should require:

1. Correct prompts beat shuffled and null prompts on retrieval/alignment with confidence above
   chance.
2. Samples beat a nearest-neighbor/retrieval baseline and a repeated mean-latent baseline on a
   distribution metric.
3. Generated renders preserve nontrivial anisotropy and mixed spacetime tilt.
4. A 120 → 600 (then 1,200) example curve improves both perceptual quality and prompt selectivity.

This separates the decisive question—whether text can predict the learned splat manifold—from
encoder redesign. If the direct prior fails after convergence while the scaffold route continues
to work, scale paired prompt/video diversity before changing the representation.

## Reproducible evidence

- Encoder curve and audits: `sol/results/encoder_convergence_v2_continued`
- Support-correct direct-fit teachers: `sol/results/support_correct_encoder_teachers_v1`
- Encoder renderer benchmark: `sol/results/encoder_support_benchmark_v1`
- Replicated temporal-tilt intervention: `sol/results/temporal_tilt_replication_v1`
- Support renderer scale gate: `sol/results/support_renderer_benchmark_v1`

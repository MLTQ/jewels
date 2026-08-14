# Domain-matched realizer: training on teacher-generated LTX fits

The twelve LTX training clips, fitted overnight at the frozen 72k/49-frame contract (replay mean
31.14 dB, `corpus/ltx_scaffold_v1_train_72k`), replace the twelve real UCF sources as the
realizer's training corpus. The split is physically disjoint and honestly labelled
(`sol/build_ltx_domain_train.py`: `validation_is_unseen=true`, `source_overlap=false`); prompt
embeddings are rebound bit-identically; recipe, seeds, rollout protocol, and the LPIPS battery
are unchanged from every other 12-source point.

## Head-to-head at 12 sources (four held-out LTX evaluation scaffolds)

| Arm | Held-out velocity loss | LPIPS | PSNR | SSIM |
|---|---:|---:|---:|---:|
| UCF-trained transfer (72 views) | 1.5582 | 0.6971 | 14.142 | 0.6001 |
| **LTX domain-matched (36 views)** | **1.0448** | **0.6760** | 12.557 | 0.5641 |
| Guide upsample baseline | — | 0.6761 | 19.347 | 0.8596 |
| Fitted jewel ceiling | — | 0.0982 | 27.521 | 0.9831 |

Per-class LPIPS (domain vs UCF arm): Basketball 0.7487 vs 0.7346; HorseRiding **0.6386** vs
0.7100; PlayingGuitar 0.6803 vs 0.6801; ApplyEyeMakeup **0.6363** vs 0.6635. Correct-shuffled
velocity margin is preserved (0.0174).

## Reading

1. **The cross-domain transfer penalty was enormous.** Same recipe, same evaluation, half the
   training views: held-out velocity loss falls 33% (1.558 → 1.045) purely from training in the
   teacher's domain. Every earlier realization number was paying an unmeasured domain tax.
2. **First macro LPIPS parity with the blur baseline** (0.6760 vs 0.6761), with clear wins on
   HorseRiding and ApplyEyeMakeup. Combined with the data-scaling curve's unsaturated slope,
   the perceptual crossover is now a data-quantity question.
3. **PSNR/SSIM drop while LPIPS improves** — the third independent occurrence of the
   blur-rewarding pattern (`../guide_upsample_baseline/`, `../data_scaling_v1/`). Reference
   metrics and perceptual metrics now disagree consistently in the direction the theory
   predicts; conclusions rest on the perceptual axis plus the fitted ceiling's 0.098 headroom.
4. **Everything in this arm's supervision is teacher-generated.** Prompt-matched videos
   (~3 min each), fits (~25 uncontended GPU-min each), training (minutes): quality in this lane
   is priced purely in compute, which is the donated-compute conversion claim made concrete.

## Visual audit — the decisive caveat

The frame-32 contact sheet (`rollout_contact.png`) fails the visual gate: the domain-matched
arm loses the recognizable macro-layout (court organization, horse/rider silhouette, guitar
axis, face structure) that the UCF-trained arm's contacts retain, dissolving into locally rich
but globally unstructured texture. Patch-based AlexNet LPIPS evidently rewards the improved
local color/texture statistics while being nearly blind to the layout regression — so this
record establishes domain matching at the feature and perceptual-statistic level only. The
33% velocity-loss improvement is real and the compute economics stand, but "better LPIPS" here
does not mean "looks better", and no visual improvement is claimed. Candidate causes for the
layout loss (36 views at 12,000 steps = 333 steps/view versus UCF's 167; 49-frame fields
halving temporal context; different jewel statistics in LTX fits) are follow-up work.

Further caveats: one seed, four evaluation clips, Basketball regresses on LPIPS too (its
scaffold is the weakest fit at 27.57-28.18 dB replay), and the domain-matched corpus carries
half the views of the UCF corpus.

Artifacts: `mark_summary.json`, `rollout_summary.json`, `perceptual_report.json`,
`rollout_contact.png`; Aine: `topology/ltx_domain_v1/`.

# Compute-feasibility gate for promptable text-to-video jewels

## Target claim

The pitch does not need a production text-to-video system. It needs evidence that a promptable
model built around time-distorted Gaussian splats has a functioning end-to-end path and improves
predictably with additional data/model/optimization compute.

The defensible claim is:

> A text-conditioned model can produce recognizable, prompt-selective videos through a persistent
> spacetime-splat representation; the representation earns quality per element from mixed
> space/time geometry; and measured learning curves have not saturated at available compute.

## Evidence earned on this branch

| Evidence | Result | Label |
|---|---|---|
| Production culling counterexample | 64-center-KNN drops a time-tilted splat contributing >0.8; support mode matches the all-splat oracle | Implementation proof |
| Synthetic corrected compute curve | 32.06 → 48.26 → 58.01 dB at 100/300/900 steps | Single-source scaling signal |
| Real corrected compute curve | 24.67 → 30.04 → 36.28 dB; median anisotropy 1.02 → 1.32 → 2.22 | Single-source scaling signal |
| Synthetic temporal-tilt intervention | Free beats axis-aligned by 0.97 and 1.05 dB at equal counts | Single-source causal signal |
| Real temporal-tilt intervention | Free 36.28 dB vs axis-aligned 34.70 dB at equal count/runtime | Single-source causal signal |
| Multi-source temporal-tilt replication | 9/9 wins; +0.772 dB paired mean, 95% CI [+0.573, +0.971], matched counts/bytes | Decision-grade representation evidence |
| Sparse support-complete renderer | Exact pixels/gradients; 1.953×/1.965×/1.990× KNN step time at 10k/45k/72k on RTX 4090 | Implementation scaling evidence |
| Balanced amortized-encoder curve | 21.876 → 22.399 → 22.628 dB at 12/60/120 examples; every largest-budget seed beats every middle-budget seed | Decision-grade bottleneck scaling evidence |
| Full-frame perceptual/structure audit | LPIPS 0.4728 → 0.4651 → 0.3916; mixed tilt 0.0479 → 0.0725 → 0.1875 | Decision-grade bottleneck evidence |
| Prompt semantic retention | Correct beats shuffled 12/12; +0.1913 cosine margin; 91.4% source alignment retained | Scaffold-path evidence |

These results establish that the corrected representation and shared video-to-splat bottleneck work
and benefit from compute. They establish prompt retention through a pretrained video scaffold, but
do not yet establish direct prompt-to-splat generation or broad generalization.

## Pitch gate

All five gates below should pass before presenting “more compute can make this” as the main claim.

### G1 — representation causality, replicated

- Run support-correct free versus axis-aligned fitting on at least three seeds and three
  motion-diverse clips.
- Free geometry wins in at least seven of nine matched pairs and by at least 0.5 dB macro PSNR.
- Report mixed-tilt distributions, quality per primitive/byte, local error, and confidence intervals.

**Status: passed.** Three UCF motion classes × three seeds produce 9/9 wins, +0.772 dB macro
advantage, a wholly positive paired interval, matched 791-primitive/72,784-byte budgets, and larger
error reduction in high-motion regions. Evidence: `sol/results/temporal_tilt_replication_v1`.

### G2 — support-safe throughput

- Replace the all-center conservative search with a support-complete tile/bin implementation.
- Match the reference renderer within the declared five-sigma truncation tolerance and preserve
  fatal overflow or an equivalent proof of completeness.
- Reach within 2× of KNN fitting throughput at the intended 10k–72k primitive regime. The current
  corrected Python/PyTorch path is 5–9× slower and is evidence code, not a corpus engine.

**Status: passed on the target RTX 4090.** The support-complete multilevel index uses rotated-
ellipsoid AABBs and an exact detached pre-gather test, matches all-center outputs and every parameter
gradient, and preserves loud overflow plus the time-tilted counterexample. Median forward/loss/
backward ratios are 1.953×, 1.965×, and 1.990× KNN at 10k, 45k, and 72k, with 8.01 GB peak allocation
at 72k on the 24 GB device. A matched 100-step real-video fit reaches the oracle's PSNR within
0.000014 dB while taking 0.69 versus 3.73 seconds. Evidence:
`sol/results/support_renderer_benchmark_v1` and `sol/results/support_tiled_e2e_v1`.

### G3 — amortized jewel production scales

- Build a support-correct teacher subset and train the video-to-jewel encoder at three data/compute
  budgets to convergence, not a shared arbitrary step count.
- Held-out support-rendered quality, LPIPS-class perceptual quality, and structural fidelity must
  improve at the last two budgets; anisotropy and mixed tilt must not collapse toward lattice dots.
- Preserve the successful video-seeded initialization as an existence proof, while retesting prior
  negative architectural conclusions under the project evidence policy.

**Status: passed at 12/60/120 examples.** Across three seeds, mean held-out PSNR rises at both
budget increases and every 120-example seed beats every 60-example seed. LPIPS falls 15.8% from 60
to 120 while layout fidelity, anisotropy, and mixed spacetime tilt remain nontrivial or improve.
Five support-correct direct-fit teachers establish additional headroom at 27.699 dB and LPIPS
0.1720. Evidence: `sol/results/encoder_convergence_v2_continued` and
`sol/results/support_correct_encoder_teachers_v1`.

### G4 — prompt selectivity survives rendering

- Use the lowest-risk end-to-end path first: prompt → pretrained video scaffold → jewel
  encoder/refiner. A later direct prompt-to-jewel prior can replace the scaffold.
- On at least 12 held-out prompts, compare correct, shuffled, and null text with fixed seeds and
  matched generation settings.
- Correct prompts must win on prompt retrieval/semantic alignment with a confidence interval above
  chance, while rendered videos remain recognizably above the trivial scaffold/blur controls.
- Previously documented prompt-prior failures are hypotheses about those configurations, not laws;
  rerun only the highest-value factorized or hierarchical variants with adequate convergence.

**Status: passed narrowly for semantic retention through the pretrained-scaffold route; direct
generation remains open.** On 12 held-out photoreal action prompts, the largest encoder's rendered
output matches correct text better than shuffled text in 12/12 cases, with a +0.1913 mean cosine
margin and 91.4% retention of source-video alignment. This localizes the next risk to predicting
the latent from text, rather than semantic destruction by the encoder/renderer. Evidence:
`sol/results/encoder_convergence_v2_continued/prompt_smoke`.

### G5 — extrapolatable scaling curve

- Measure at least three increasing data/model/optimization budgets for the end-to-end promptable
  path.
- Fit and publish the learning curve with uncertainty. The largest two points must still improve on
  semantic selectivity and perceptual quality.
- State the compute multiplier projected to reach the pitch quality bar. A credible bounded
  extrapolation, plus qualitative videos, is the requested feasibility proof.

## Immediate sequence

1. Freeze the selected 120-example encoder and train a prompt-conditioned flow/diffusion model over
   its structured splat latent, with correct/shuffled/null controls and rendered evaluation.
2. Scale the balanced corpus to 600 and then 1,200 examples, retaining held-out source ownership and
   replicated seeds at the largest decision points.
3. Distill direct-fit teacher geometry and add sparse slot/count prediction to narrow the LPIPS and
   mixed-tilt gap without relying on all 73,728 slots.

Interpret every step under `sol/EVIDENCE_POLICY.md`. A failed arm narrows a configuration; it does
not retire a mechanism without the required replication and convergence checks.

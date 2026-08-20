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

These results establish that the corrected stage-1 mechanism works and benefits from compute. They
do not yet establish promptability or broad generalization.

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

### G4 — prompt selectivity survives rendering

- Use the lowest-risk end-to-end path first: prompt → pretrained video scaffold → jewel
  encoder/refiner. A later direct prompt-to-jewel prior can replace the scaffold.
- On at least 12 held-out prompts, compare correct, shuffled, and null text with fixed seeds and
  matched generation settings.
- Correct prompts must win on prompt retrieval/semantic alignment with a confidence interval above
  chance, while rendered videos remain recognizably above the trivial scaffold/blur controls.
- Previously documented prompt-prior failures are hypotheses about those configurations, not laws;
  rerun only the highest-value factorized or hierarchical variants with adequate convergence.

### G5 — extrapolatable scaling curve

- Measure at least three increasing data/model/optimization budgets for the end-to-end promptable
  path.
- Fit and publish the learning curve with uncertainty. The largest two points must still improve on
  semantic selectivity and perceptual quality.
- State the compute multiplier projected to reach the pitch quality bar. A credible bounded
  extrapolation, plus qualitative videos, is the requested feasibility proof.

## Immediate sequence

1. Refit a small support-correct teacher subset with the tiled renderer and run the
   convergence-based encoder scaling curve for G3.
2. Assemble the safest promptable demo through a pretrained scaffold, then measure correct versus
   shuffled/null prompt selectivity at the rendered jewel output.
3. Only after that end-to-end signal exists, compare direct hierarchical/factorized prompt-to-jewel
   priors and scale the winning route.

Interpret every step under `sol/EVIDENCE_POLICY.md`. A failed arm narrows a configuration; it does
not retire a mechanism without the required replication and convergence checks.

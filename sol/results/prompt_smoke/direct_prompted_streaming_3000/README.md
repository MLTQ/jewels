# Direct-jewel prompted streaming smoke

## Decision

The learned reconstruction tokenizer remains below the visual gate even after a substantial metric
gain. Prompt conditioning therefore moves to the already validated persistent continuation path,
which predicts canonical 22-D jewel births directly and copies carried jewels exactly.

This is the first run in the project where a held-out text prompt changes independently decoded
jewel topology in the expected class-specific direction. It is a semantic-selectivity result, not
yet a recognizable text-to-video result. A subsequent controlled audit rejects the proposed
occupied-cell/positive-count split as the remedy for washed-out video: exact target topology barely
changes the deterministic render. The supported next architecture is a stochastic jewel realizer
conditioned on a coherent video scaffold.

## Why the tokenizer path paused

The exposure-matched shared `64^3 × 8` tokenizer completed 12,000 steps in 73.3 minutes. Its fixed
4,096-point group-4 audit reaches **20.735 dB / 99.794% count**, while four seen group-1 sources
reach **22.413 dB / 98.405% count**. Both sets still dissolve actors into mottled texture, proving
that data generalization is not the only failure.

Occupied-group controls preserve count as explicit sparse topology and compare matched latent
budgets on the HorseRiding overfit:

| Codec | Fixed PSNR | Count | Interpretation |
|---|---:|---:|---|
| Dense `64^3 × 8`, balanced count | 24.586 dB | 97.69% | Previous best |
| 4 jewels/token × 32-D | 24.752 dB | 100% | Sparse topology works; weak render objective |
| 2 jewels/token × 24-D | 24.317 dB | 100% | Lower feature error does not imply better fields |
| 4 jewels/token × 32-D, render weight 2 | **26.019 dB** | 100% | +1.43 dB over previous best |
| 1 jewel/token × 16-D, render weight 2 | **26.087 dB** | 100% | Only +0.07 dB over four-jewel tokens |

Even the one-jewel control remains visibly noisy. Tiny independent covariance and opacity errors
accumulate across 120,000 Gaussians, so compressing and reconstructing jewels before generation is
the wrong critical path. Sparse addressed jewels should be the model tokens; learned hierarchy can
be added above them without introducing a lossy round-trip.

## Prompted direct-birth protocol

- Training: 12 UCF source groups, four actions, 48 continuation views.
- Validation: all four group-4 videos and their unseen prompt templates, 16 views.
- Representation: exact carried jewels plus learned frontier-local 22-D births on a `16×16×8`
  grid with a measured-safe 512-rank capacity.
- Model: 0.84M parameters, local prefix raster, frozen 512-D text embedding.
- Training controls: 15% text dropout for a learned null branch; 25% context dropout in the first
  3,000-step run so text-only behavior is trained rather than assumed.

The 3,000-step run finishes in 25.2 seconds. On held-out videos:

| Text-only control | Birth-feature MSE | Count MAE/cell |
|---|---:|---:|
| Correct unseen template | **1.1579** | **5.155** |
| Shuffled action | 1.1794 | 5.279 |
| Null text | 1.1641 | **5.125** |

Correct text beats shuffled text on both targets, while null slightly beats correct density. Free
decoding reveals deterministic class-specific prompt counts: Basketball 11,420; HorseRiding
10,790; PlayingGuitar 9,141; ApplyEyeMakeup 9,081; null 10,140. Swapping text swaps those counts,
and correct text beats shuffled text in low-resolution field PSNR for all four held-out classes.

The videos remain washed out. They establish prompt-to-density and a small prompt-to-mark signal,
not visible action generation.

## Follow-up controls

A fresh 12,000-step run with 50% context dropout gives each training view about 125 text-only
updates. Correct held-out text improves mark MSE to **1.1585 versus 1.2029 shuffled** (3.69% better)
and 1.1658 null, but its density is worse than null. A 6,000-step occupied/empty-balanced count
control further worsens overall count MAE by trading missed occupied cells for false positives.

These runs motivated a topology-head redesign, but the washout decomposition below supersedes that
plan. Count calibration remains necessary for autonomous generation; it is not the current visual
bottleneck.

## Washout decomposition: topology is not the cause

The audit holds the target birth cells, counts, ranks, carried state, render grid, and prompt fixed,
then substitutes predicted mark groups. Across all four group-4 holdouts:

| Controlled output | Mean field PSNR | Interpretation |
|---|---:|---|
| Free deterministic prediction | 14.060 dB | Count and marks predicted |
| Predicted marks, exact target topology | 14.232 dB | Only +0.172 dB; still visibly washed out |
| Predicted geometry only | 14.434 dB | Geometry errors independently destroy structure |
| Predicted color only | 15.255 dB | Color/gradient regression independently washes appearance |
| Predicted opacity only | 24.816 dB | Opacity is not the dominant failure |

Only 39.73% of deterministic predicted centers remain in their declared spatial cell and 25.41%
remain in their full birth cell, even though that cell is supplied to the model. This reveals a
contract weakness, but exact topology does not repair the picture. The main failure is deterministic
Smooth-L1 regression over a permutation-ambiguous, one-to-many set of 22-D marks: it produces a
conditional average of incompatible positions, covariances, and colors.

## Stochastic mark-flow and semantic-scaffold controls

A 1.54M-parameter rectified flow was trained for 12,000 steps with exact target topology. It restores
detail energy but not coherent content: raw samples average 15.064 dB and 84.44% of target edge
energy, yet the GIFs are structured speckle rather than recognizable action. Stochasticity is
necessary, but a local prefix and four-class text embedding do not specify the missing global scene.

The decisive control adds the true future video only as a `24x40` low-resolution guide raster to a
2.13M-parameter mark flow. It is an oracle experiment, not an inference-time result. With the same
target topology and held-out sources, the guided jewelizer reaches:

| Mark generator | Mean PSNR | Contrast ratio | Edge-energy ratio |
|---|---:|---:|---:|
| Deterministic marks | 14.232 dB | 0.574 | 0.598 |
| Unguided stochastic flow, raw | 15.064 dB | 0.775 | 0.844 |
| Oracle-video-guided stochastic flow | **16.555 dB** | **0.856** | **0.905** |

The corrected hard topology projection leaves every fitted target render-identical at 100 dB, so
the +2.324 dB guided gain over deterministic marks is not a projection artifact. The guide restores
the court/player, horse/rider, guitar/player, and face/hand macro-layout; residual high-frequency
noise shows that the current cell-mean guide encoder and feature-only flow loss are not yet a
high-quality jewelizer. Correct and shuffled text remain similar because the oracle future guide
already supplies nearly all scene semantics.

## Architecture decision

Do not train the occupied-cell/count head next. The best-supported path is:

1. use a pretrained text-to-video model to generate a coherent, low-resolution semantic scaffold;
2. train a multiscale, stochastic video-to-jewel realizer with spatial cross-attention and direct
   differentiable render/perceptual supervision;
3. once oracle-guide jewels are coherent, learn conditional birth occupancy/count around that
   scaffold and preserve exact carried jewels across overlapping windows;
4. distill away the raster scaffold only after a much larger captioned jewel corpus exists.

This hybrid does not abandon native jewels: the final persistent, selectable, movable, and locally
repairable state remains the jewel field. It uses mature raster video priors for the semantic task
that 12 training clips cannot supply.

## Artifacts

- `summary.json`, `train_log.jsonl`: 3,000-step optimization and held-out controls.
- `visual_report.json`: free-count birth totals and field PSNR for all four held-out classes.
- `*_prompt_controls.gif`, `*_contact.png`: fitted target, prefix+correct, and text-only
  correct/shuffled/null comparisons.
- `context50_12000/*`: higher text-only-exposure run.
- `balanced_6000/*`: occupied/empty-balanced count negative control.
- `washout_audit/*`: exact-topology and mark-group decomposition GIFs plus metrics.
- `mark_flow_12000/*`: unguided oracle-topology stochastic-flow training and visual controls.
- `mark_flow_oracle_guide_12000/visual_contract_projection/*`: authoritative oracle-video-guide
  GIFs, contact sheets, and metrics after target-preserving topology projection.

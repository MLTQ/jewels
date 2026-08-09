# Direct-jewel prompted streaming smoke

## Decision

The learned reconstruction tokenizer remains below the visual gate even after a substantial metric
gain. Prompt conditioning therefore moves to the already validated persistent continuation path,
which predicts canonical 22-D jewel births directly and copies carried jewels exactly.

This is the first run in the project where a held-out text prompt changes independently decoded
jewel topology in the expected class-specific direction. It is a semantic-selectivity result, not
yet a recognizable text-to-video result.

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

The next topology head must factorize:

1. occupied-cell probability with a calibrated sparse classification loss;
2. positive count conditioned on occupancy;
3. direct jewel marks conditioned on the emitted topology, local prefix, and text.

Only after free decoded renders show visible correct-versus-shuffled action differences should we add
an initial-window generator and autonomous multi-stride rollout.

## Artifacts

- `summary.json`, `train_log.jsonl`: 3,000-step optimization and held-out controls.
- `visual_report.json`: free-count birth totals and field PSNR for all four held-out classes.
- `*_prompt_controls.gif`, `*_contact.png`: fitted target, prefix+correct, and text-only
  correct/shuffled/null comparisons.
- `context50_12000/*`: higher text-only-exposure run.
- `balanced_6000/*`: occupied/empty-balanced count negative control.

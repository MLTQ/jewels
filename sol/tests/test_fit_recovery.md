# test_fit_recovery.py

## Purpose

Prove that a long stage-1 primitive fit can be interrupted after an atomic recovery checkpoint
and resume to the same numerical result as an uninterrupted run.

## Coverage

### `test_resume_matches_uninterrupted_across_densification`

- Runs a seven-step deterministic CPU control fit and, when CUDA is present, repeats on the last
  visible GPU (the allocated server maps this to the 2070S).
- Runs the same configuration again and deliberately interrupts after step 3, immediately after
  serializing the fitter's `next_step == 4` recovery boundary.
- Places the checkpoint after densification at step 2 and after one subsequent Adam update. This
  verifies a changed primitive count, non-empty optimizer moments, a partially accumulated
  gradient tracker, history, background, and RNG state—not merely field weights.
- Loads the atomic file and resumes through another densification event.
- Requires every final field tensor and all non-timing metadata to equal the uninterrupted control.

## Contracts

| Dependency | Assumption |
|------------|------------|
| `fit.fitter.fit_volume` | Recovery callbacks describe the next step after a complete update |
| `fit.recovery.atomic_torch_save` | Replaced file is immediately loadable as a restricted tensor payload |

Timing is excluded because wall-clock duration necessarily changes across a stopped process.

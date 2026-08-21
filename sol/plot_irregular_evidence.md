# `plot_irregular_evidence.py`

## Purpose

Turns the compact, source-owned JSON experiment logs into a pitch-readable view of the fidelity,
grid-suppression, sparsity, and mixed-spacetime-tilt tradeoffs.

## Components

### `evaluation_rows`
- **Does**: Reads held-out evaluation records only, avoiding the incomparable minibatch training
  metric, and normalizes each record into stable scalar fields.

### `summary_point`
- **Does**: Extracts the held-out PSNR/occupancy pair used by the matched 200-step causal control.

### `main`
- **Does**: Draws the frozen PSNR, occupancy, active-fraction, and mixed-tilt thresholds.
- **Does**: Contrasts the appearance-biased and direct-tilt trajectories and shows that no current
  run occupies the desired high-quality/low-uniformity quadrant.
- **Does**: Offsets the appearance-only continuation by its source checkpoint's 2,000 steps so its
  flat geometry and saturating fidelity are compared on total-compute rather than local-step axes.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Evidence report | JSONL evaluation rows contain `macro_psnr` and `structure` | Log schema |
| Matched control | Summary files describe the same held-out five-source corpus | Dataset ownership |
| Gate interpretation | Dashed thresholds match `PROTOCOL.md` | Gate revision |

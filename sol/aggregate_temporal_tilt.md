# `aggregate_temporal_tilt.py`

## Purpose

Combines multi-seed, multi-source temporal-tilt ablations into the decision-grade replication gate
defined by `COMPUTE_FEASIBILITY_GATE.md`.

## Components

### `aggregate_reports(reports)`
- **Does**: validates a common v2 protocol, collects the largest-budget free/axis-aligned pairs,
  reports per-source and aggregate paired uncertainty, and evaluates every predeclared gate.
- **Interacts with**: `mean_confidence_interval` and `write_report` from
  `temporal_tilt_ablation.py`.
- **Rationale**: aggregation is separate from GPU fitting so completed source reports remain
  independently auditable and can be recombined without rerunning fits.

### `main()`
- **Does**: loads report paths, writes `temporal-tilt-replication-v1`, and prints the paired effect.

## Decision gate

- At least three distinct sources and nine paired seeds.
- Free geometry wins at least seven pairs and by at least 0.5 dB macro PSNR.
- The paired 95% Student-t interval excludes zero.
- Primitive counts and raw parameter bytes are matched in every pair.
- Axis-aligned projection removes mixed spacetime tilt in every control.

PSNR-per-primitive and PSNR-per-byte ratios are descriptive quality-at-budget summaries; the
causal comparison remains the paired PSNR difference at exactly matched budgets.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Replication evidence | `temporal-tilt-replication-v1` with individual pairs and gate booleans | Schema or gate definitions |
| `temporal_tilt_ablation.py` | v2 protocol and comparison fields | Record or comparison schema |

# `frontier_contribution_loss.py`

## Purpose

Targets the diagnosed blank-frame ramp directly by supervising how strongly newly emitted jewels
can contribute at local frontier time zero, without broadly retuning color or motion rendering.

## Components

### `frontier_peak_alpha`

- **Does**: Recovers marginal temporal sigma in bounded chunks and computes each jewel's peak alpha
  on the frontier frame plane.
- **Interacts with**: `temporal_standard_deviation` in `splat_density.py`.
- **Gradient boundary**: Sigma is detached because repeated covariance eigenvalues make eigenvector
  derivatives undefined; temporal centers and opacity still receive the intended correction.

### `frontier_contribution_loss` / `FrontierContributionLoss`

- **Does**: Matches square-root per-jewel alpha, log cell alpha mass, and a differentiable soft count
  above the declared 5% visibility threshold.
- **Rationale**: Lifecycle activity alone counts weak 3-sigma tails; the renderer needs thousands of
  materially contributing jewels at frame zero.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Scaffold mark fine-tune | Features are frontier-local and topology rows are target-aligned | Time/row semantics |
| Density audit | Default visible threshold matches the reported 5% peak-alpha count | Threshold |
| 2070S training | Covariance eigensolves stay bounded by `covariance_chunk` | Chunking |

## Notes

- Cell pooling retains the full `(u,v,birth-time)` address. It does not invent correspondence
  between different stochastic topologies; this is a target-topology training loss only.

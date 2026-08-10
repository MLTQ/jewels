# `audit_prompted_washout.py`

## Purpose

Determines whether prompted-continuation washout comes primarily from decoded birth topology or
from the predicted Gaussian marks. It uses target-rank hybrids so a count-head change is not blamed
for errors that remain under oracle topology.

## Components

### `replace_groups`
- **Does**: Builds aligned hybrids by copying canonical center, covariance, color, gradient, or
  opacity groups between target and predicted jewels.
- **Rationale**: Rendering one predicted group at a time exposes nonlinear compositing failures that
  per-feature MSE cannot localize.

### `topology_adherence`
- **Does**: Measures whether predicted jewel centers remain in their assigned spatial cells and
  whether their covariance-derived first-active frame remains in the assigned birth cell/stride.
- **Interacts with**: `to_global_time` and the finite-support lifecycle convention.
- **Rationale**: A count attached to a raster cell is not meaningful if its decoded mark migrates
  elsewhere or becomes active in a different window.

### `render_signature`
- **Does**: Reports PSNR and target-relative contrast, edge, saturation, and temporal-change energy.
- **Rationale**: PSNR alone can reward a low-detail conditional mean despite visible washout.

### `main`
- **Does**: Audits every held-out action using correct prefix and text, renders carried-only,
  oracle-topology, group-hybrid, and free-count fields, and writes GIF/contact/JSON artifacts.
- **Interacts with**: `BirthContinuationModel`, the prompt corpus, and the exact jewel renderer.
- **Memory contract**: Runs the batched symmetric local/global covariance transform on CPU before
  returning features to the requested renderer device; some CUDA eigensolvers reserve workspace
  far larger than these small matrices warrant.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Architecture decision | Oracle predicted marks use exact target cells, ranks, and birth count | Panel construction |
| Lifecycle audit | Birth time is derived from temporal support, not Gaussian center alone | Support convention |
| Visual gate | Every field uses the same carried state, points, background, and target | Render provenance |

## Notes

- The audit diagnoses the current deterministic model; it does not claim that target-rank
  correspondence is suitable for a stochastic generator.
- Correcting occupancy/count prediction can only cure washout if oracle-topology predicted marks
  already retain target detail.

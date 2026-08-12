# `birth_mark_flow.py`

## Purpose

Defines the stochastic jewel-mark generator selected after oracle decomposition showed that count
calibration cannot cure prompted washout. Cell occupancy and canonical ranks remain external so the
first experiment isolates geometry/appearance generation from topology.

## Components

### `rasterize_noisy_marks`
- **Does**: Pools noisy per-jewel mean, variance, count, and occupancy into the declared birth grid.
- **Rationale**: The velocity of one jewel must see correlated noisy state in neighboring cells;
  independent per-rank denoising would merely replace blur with incoherent speckle.

### `BirthMarkFlowModel`
- **Does**: Combines exact prefix rasters, noisy-set raster context, cell/rank identity, flow time,
  optional aligned cell-raster or multiscale token video guidance, text, and each noisy mark to
  predict a 22-D rectified-flow velocity.
- **Interacts with**: `ContextRasterEncoder`, canonical rank features, and prompt embeddings.
- **Rationale**: Noise makes the one-to-many continuation distribution representable instead of
  forcing Smooth-L1 regression to its conditional mean.
- **Multiscale path**: Mean-pooled token features enter the cell field, then each cell/rank/noisy
  mark query cross-attends only to tokens owned by its addressed cell. This preserves bounded cost
  while allowing ranks in one cell to bind to different scaffold edges and colors.
- **Hybrid path**: Raster and token guidance may be enabled together. The raster encoder supplies
  cross-cell 3D convolution/global context while token attention supplies within-cell detail.

### `birth_mark_flow_objective`
- **Does**: Scores an explicit Gaussian-noise-to-target flow path with topology fixed.
- **Rationale**: Fixed paths support correct/shuffled/null comparisons without sampling variance.
- **Conditioning**: Accepts either a real text embedding or the model's trained null branch.

### `sample_birth_marks`
- **Does**: Euler-integrates one variable-size mark set and optionally applies classifier-free text
  guidance.

### `project_birth_topology`
- **Does**: Constrains sampled spatial centers to assigned cells and shifts temporal centers so the
  finite-support start lies inside the assigned birth-time cell.
- **Rationale**: The previous direct decoder let 57–63% of held-out centers escape their topology;
  addressed cells must be a hard generative contract.
- **Discrete-time contract**: Uses the full continuous support-start interval that rounds up to a
  physical first-active frame in the assigned bin, so already valid target jewels are unchanged.
- **Boundary cells**: Spatial edge cells retain their unbounded exterior half-space because
  `GridSpec.cell_index` assigns out-of-frame centers there by clamping the address, not the center.
- **Gradient contract**: Projection is out-of-place so differentiable render supervision can
  backpropagate through projected denoised marks. Temporal support extent is detached only while
  calculating the hard center boundary, avoiding undefined eigenvector gradients at repeated
  covariance eigenvalues; covariance parameters still receive gradients through rendering.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Prompted mark-flow trainer | Standardized target marks and one variable-size topology | Feature units |
| Oracle visual gate | Counts/cells/ranks are target-owned; only marks are sampled | Ownership split |
| Oracle video-guide gate | `guide_dim=3` accepts canonical `(cells,RGB)` future scaffolds | Guide shape |
| Multiscale realizer | `guide_token_dim>0` accepts `(cells,tokens,features)` scaffolds | Token shape |
| Future topology model | Emits cells/ranks consumed unchanged by this model | Index convention |
| Streaming renderer | Projection returns frontier-local canonical 22-D features | Coordinate semantics |

## Notes

- This is intentionally not a complete generator: oracle topology is a scientific control.
- Mark-flow success is required before learning a stochastic occupancy/count process.
- The local noisy raster provides bounded correlation at 20k-jewel scale without quadratic global
  point attention.

# `render_scaffold_mark_rollout.py`

## Purpose

Executes and visualizes fitted-seed-free jewel sequences: initial generation plus two free-running
continuation strides under correct, shuffled, and null video scaffolds. It also runs the optional
two-stream lifecycle/appearance control or compact RGB-adapter control against a matched frozen
base, and can isolate an augmented mark model under an exact base-owned topology sequence.

## Components

### `_causal_background`

- **Does**: Uses only the first scaffold stride's RGB mean as one persistent background color.
- **Rationale**: This removes the fitted background privilege with a transparent baseline before
  attempting to learn a decomposition-gauge parameter from only 12 training videos.

### `_seam_report` / `_density_report`

- **Does**: Measures window-boundary change relative to normal motion and target seams, plus
  contribution-aware effective/visible splats per frame. Density reports retain every initial
  stride value and the exact value at each later frontier so ramp-in cannot hide in a mean.
- **Saliency audit**: The final report also includes target-derived foreground RGB/edge,
  motion-boundary, and quiet-region temporal errors from `saliency_metrics.py`.

### `main`

- **Does**: Reconstructs checkpoint contracts, runs end-to-end correct/cross-class/null rollouts,
  renders 48 frames, compares against the LTX scaffold and fitted ceiling, and saves editable jewel
  fields, GIFs, contact sheets, and a JSON report.
- **Control isolation**: Text, random seed, model, and causal background remain fixed; only the
  scaffold sequence changes. Later carry is free-running within each control.
- **Repeatable ablations**: `--deterministic` enables deterministic PyTorch/cuDNN algorithms and a
  cuBLAS workspace contract before model creation. This is required for checkpoint comparisons
  because tiny scatter differences can alter later generated carry and amplify across windows.
- **Source-stable seeds**: Each source adds its index in the complete sorted validation split to the
  declared base seed. Filtering expensive high-resolution runs therefore cannot silently change a
  class's Gaussian path; the realized seed is stored in every source record.
- **Boundary control**: `--strict-initial-boundary` disables the left-censored clip-start exception
  while leaving every continuation frontier strict in both arms. The summary records the active
  boundary contract.
- **Provenance**: The summary records the sampling seed, PyTorch/device runtime, every checkpoint
  and corpus input path, and the prompt-manifest digest so future matched controls are reproducible.
  CUDA reports include the physical GPU name and compute capability because deterministic kernels
  do not imply bit-identical free-running trajectories across GPU architectures.
- **Two-stream gate**: `--appearance-flow` integrates the selected `--mark-flow` in lockstep,
  renders it beside the constrained candidate, and records exact lifecycle, topology, and ID audits
  for all three scaffold controls.
- **Compact adapter gate**: `--appearance-adapter` verifies the adapter's frozen-base hash, RGB-only
  dimensions, standardizers, grid, and trained saliency fraction before running the same paired
  correct/shuffled/null audit.  The two appearance mechanisms are mutually exclusive.
- **High-resolution paired gate**: `--correct-only` retains fitted ceiling, frozen base, and correct
  generation while skipping shuffled/null rollout and rendering.  It is reserved for expensive
  native-aspect paired metrics after causal scaffold separation has already been established.
- **Base-owned topology gate**: `--paired-base-flow` first runs the frozen base normally, then runs
  the augmented candidate with a fresh identically seeded generator and the base's exact per-cell
  counts/ranks in every window. Shared checkpoint tensors must be bit-identical. Candidate marks
  still create their own causal context/carry, isolating mark coupling without count feedback.
- **Dimension screens**: `--appearance-dimension-set` applies named, checkpoint-independent masks
  after every sampler/projection/coordinate-transform stage, localizing whether geometry, opacity,
  RGB, or spatial/temporal gradients own an improvement or regression.
- **Spatial residual gate**: `--appearance-saliency-fraction` enables residuals only for the top
  guide-derived foreground/motion/chroma/boundary cells in each stride. It uses the causal initial
  scaffold background and never reads evaluation masks or fitted jewels.
- **Residual calibration**: `--appearance-strength` scales the selected residual in `[0,1]` without
  changing its cells or feature dimensions.  The realized mean strength is serialized per window,
  allowing a conservative quiet-stability calibration to remain explicit and reproducible.

### `_panel_names`

- **Does**: Preserves the five-panel baseline artifact or inserts the frozen-base panel for a
  lifecycle/appearance experiment.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| P1 generation gate | No fitted marks, counts, context, carry, or background enter generated panels | Ownership |
| Visual review | Baseline keeps five panels; paired runs add the labeled frozen base | Artifact schema |
| Interactive prototype | Correct rollout saves canonical features plus stable IDs | Field schema |
| Scientific report | Global and saliency render metrics, seams, density, capacity, and controls are present | JSON schema |
| Matched fine-tunes | Deterministic mode and seed are identical between checkpoint arms | RNG policy |
| Filtered reruns | Source seed depends on full validation order, never subset order | RNG policy |
| Initial-boundary ablation | Strict and censored runs differ only at frontier zero | Projection policy |
| Result comparison | Seed, runtime, inputs, and manifest digest are explicit | Provenance schema |
| Cross-device comparison | GPU name/capability are recorded; paired deltas remain device-local | Hardware policy |
| Lifecycle factorization | Paired summaries expose exact state ownership for every control | Audit schema |
| Compact RGB adapter | Checkpoint must match the exact frozen-flow SHA and top-cell gate | Base/gate policy |
| Correct-only gate | Summary names the evaluated controls and omits no paired base metrics | Control policy |
| Spatial residual screen | Gate fraction and per-window realized strengths are serialized | Gate policy |
| Residual calibration | Summary records the global residual multiplier | Strength policy |
| Coupled-mark attribution | Shared tensors, cell counts, birth budget, and seeds are exact | Pairing policy |

## Notes

- The fitted jewel field is rendered only as a ceiling panel and evaluation reference. It is never
  passed to the topology or mark rollout.

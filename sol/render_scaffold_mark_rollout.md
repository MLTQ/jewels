# `render_scaffold_mark_rollout.py`

## Purpose

Executes and visualizes fitted-seed-free jewel sequences: initial generation plus two free-running
continuation strides under correct, shuffled, and null video scaffolds. It also runs the optional
two-stream lifecycle/appearance control against a matched frozen base.

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
- **Boundary control**: `--strict-initial-boundary` disables the left-censored clip-start exception
  while leaving every continuation frontier strict in both arms. The summary records the active
  boundary contract.
- **Provenance**: The summary records the sampling seed, PyTorch/device runtime, every checkpoint
  and corpus input path, and the prompt-manifest digest so future matched controls are reproducible.
- **Two-stream gate**: `--appearance-flow` integrates the selected `--mark-flow` in lockstep,
  renders it beside the constrained candidate, and records exact lifecycle, topology, and ID audits
  for all three scaffold controls.
- **Dimension screens**: `--appearance-dimension-set` applies named, checkpoint-independent masks
  after every sampler/projection/coordinate-transform stage, localizing whether geometry, opacity,
  RGB, or spatial/temporal gradients own an improvement or regression.
- **Spatial residual gate**: `--appearance-saliency-fraction` enables residuals only for the top
  guide-derived foreground/motion/chroma/boundary cells in each stride. It uses the causal initial
  scaffold background and never reads evaluation masks or fitted jewels.

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
| Initial-boundary ablation | Strict and censored runs differ only at frontier zero | Projection policy |
| Result comparison | Seed, runtime, inputs, and manifest digest are explicit | Provenance schema |
| Lifecycle factorization | Paired summaries expose exact state ownership for every control | Audit schema |
| Spatial residual screen | Gate fraction and per-window realized strengths are serialized | Gate policy |

## Notes

- The fitted jewel field is rendered only as a ceiling panel and evaluation reference. It is never
  passed to the topology or mark rollout.

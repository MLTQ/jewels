# `render_scaffold_mark_rollout.py`

## Purpose

Executes and visualizes the first fitted-seed-free jewel sequence: initial generation plus two
free-running continuation strides under correct, shuffled, and null video scaffolds.

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

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| P1 generation gate | No fitted marks, counts, context, carry, or background enter generated panels | Ownership |
| Visual review | Five panel names and six boundary-focused contact rows remain stable | Artifact schema |
| Interactive prototype | Correct rollout saves canonical features plus stable IDs | Field schema |
| Scientific report | Global and saliency render metrics, seams, density, capacity, and controls are present | JSON schema |
| Matched fine-tunes | Deterministic mode and seed are identical between checkpoint arms | RNG policy |
| Initial-boundary ablation | Strict and censored runs differ only at frontier zero | Projection policy |
| Result comparison | Seed, runtime, inputs, and manifest digest are explicit | Provenance schema |

## Notes

- The fitted jewel field is rendered only as a ceiling panel and evaluation reference. It is never
  passed to the topology or mark rollout.

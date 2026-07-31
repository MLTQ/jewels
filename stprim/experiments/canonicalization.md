# canonicalization.py

## Purpose
The load-bearing check for stage 2: same clip, different seeds, comparable primitive sets. If
fits aren't canonical, there is no learnable distribution over primitive sets and no model
capacity fixes it.

## Components

### `chamfer(a, b)`
- **Does**: symmetric mean nearest-neighbour distance between center sets

### `marginals(field)`
- **Does**: scale median/IQR, anisotropy median, weight median
- **Rationale**: even when individual primitives don't correspond, matching *distributions* are
  enough for a permutation-invariant (set-diffusion) prior. Distinguishes "weakly canonical"
  from "hopeless".

### `main()`
- **Does**: two seeded fits, reports Chamfer + a random-cloud baseline + marginals

## Decisions
- Random-cloud baseline included because raw Chamfer is uninterpretable without a scale. The
  **ratio** is the number to read.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| stage 2 design | ratio << 1, or ratio ~1 with matching marginals | — |

## Notes
- Measured at full budget on real footage 2026-07-31: **ratio 0.621, marginals matching to
  ~2-3%** → weakly canonical. Consequences: permutation-invariant set prior is viable;
  autoregressive over the set is ruled out (no per-primitive correspondence to order by).
- A voronoi arm was measured too (0.745 vanilla; 0.684 with a background pseudo-cell; 0.784
  with Lloyd — Lloyd canonicalizes local packing but leaves lattice phase gauge-free, so
  relative canonicality *worsened*). Branch removed; full numbers in PROJECT.md.
- Lesson recorded for posterity: toy-budget ratios are meaningless — undertrained fits haven't
  converged anywhere.

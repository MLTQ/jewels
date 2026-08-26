# `audit_jewel_casting_language.py`

## Purpose

Runs Gate 0 on independently fitted irregular fields. It asks whether a learned bundle vocabulary
retains continuous rendering and spacetime structure and whether equivalent videos share motif
programs more strongly than unrelated controls.

## Components

### `load_field_records`
- **Does**: Loads source-owned fitted checkpoints through the canonical gauge-free featurizer.
- **Rationale**: Lattice encoder slots are not allowed to manufacture apparent language stability.

### `residual_metrics` / `center_irregularity`
- **Does**: Measures token-owned standardized variance, casts per window, within-cell spread, and
  explicit center locking. It independently reports source and serialized Jewel counts so the
  no-drop gate cannot pass tautologically.

### `pairwise_language_similarity`
- **Does**: Compares raw and cell-conditional motif histograms for same-source independent fits and
  different-source controls.
- **Rationale**: Primitive-level identity is non-unique; a useful language should retain stable
  local motif distributions.

### `_render` / `_audit_candidate`
- **Does**: Uses the support-complete production renderer on identical random volume points and
  reports source-relative PSNR, structural retention, irregularity, and a clearly labeled
  per-feature marginal diagnostic. The marginal diagnostic is not a primitive-correspondence
  claim and is not a gate criterion.

### `main`
- **Does**: Fits increasing vocabularies on source-disjoint fields, audits residual scales
  `0/0.5/1`, includes cell-center quantization, saves codebooks and an incremental JSON report, and
  evaluates the preregistered largest-vocabulary gate.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate-0 report | Every validation source has at least two independent fits | Equivalence grouping |
| Scientific review | Training excludes every validation source and controls share render points | Split/render ownership |
| Plotter | Schema `jewel-casting-language-gate-v0` and per-vocabulary macro/canonicality rows | Report schema |

## Notes

- Checkpoint codebooks are evidence artifacts, not promoted generator weights.
- Random-volume PSNR isolates language reconstruction from the source video's own fit error.

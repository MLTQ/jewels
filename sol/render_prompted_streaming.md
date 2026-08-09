# `render_prompted_streaming.py`

## Purpose

Turns the direct-jewel prompt metrics into free-count held-out video artifacts. It compares fitted
future fields with correct, shuffled, and null text while preserving carried jewels exactly.

## Components

### `main`
- **Does**: Restores the prompt corpus and checkpoint provenance, independently decodes birth counts
  and marks, returns frontier-local jewels to global time, merges exact carried state, and renders
  all four held-out action classes.
- **Controls**: Shows prefix-plus-correct text beside text-only correct, different-class, and null
  prompts.
- **Interacts with**: `BirthContinuationModel`, `streaming_corpus.py`, and the exact renderer.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Visual prompt gate | Correct/shuffled/null use one checkpoint and matched render grids | Panel semantics |
| Density audit | Labels and JSON record independently decoded birth counts | Count provenance |
| Persistent state | Every candidate concatenates bit-identical carried jewels | Merge semantics |

## Notes

- The text-only branch removes the prefix raster but retains the external carried state in the
  rendered field. Initial-video generation still requires a separate first-window topology model.
- Low-resolution exact rendering is a structural diagnostic, not presentation-quality video.

# `probe_cell_quantization.py`

## Purpose

Tests the precondition for the LLM-emission architecture: can a video window survive being
reduced to a short sequence of **discrete cell codes**? Individual jewels are far too numerous
to be tokens (~36k/second), but cells are not (2,048 per 2-second window), so the whole
direction rests on quantized cell latents still decoding into watchable video.

## Components

### `kmeans` / `quantize` / `split_groups`
- **Does**: Fits Lloyd codebooks on train-split cells only, assigns held-out cells to codes, and
  optionally splits the channel axis into product-quantizer groups (more codes per cell, each
  over a narrower block).
- **Rationale**: Product quantization trades sequence length against fidelity — the exact knob
  an LLM budget cares about.

### `main`
- **Does**: For style-stratified held-out windows, renders the unquantized latent (the ceiling)
  and every quantized variant through the frozen encoder, reporting render/layout signatures
  plus tokens and bits per window.
- **Rationale**: The `seed` half is deliberately left unquantized so the measurement isolates
  the cell-code question rather than confounding it with coarse-video compression.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| LLM-emission design | Schema `cell-quantization-probe-v1` with per-arm token counts | Report fields |

## Notes

- Codebooks come from the train split only; held-out windows are never fitted on.

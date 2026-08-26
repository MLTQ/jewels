# `render_individual_jewel_language.py`

## Purpose

Produces the qualitative counterpart to the passing Gate 0f metrics. It compares continuous
optimized targets with the actual claimed representation: exact irregular centroids plus only the
covariance, surface/opacity, and gradient token IDs from the frozen K=1,024 vocabularies.

## Components

### `select_records`

- **Does**: Selects the lowest fitter seed for each protocol-owned validation source in frozen
  protocol order.

### `main`

- **Does**: Encodes and decodes the active three-token Jewel representation, renders identical
  early/middle/late coordinates, and writes a labeled source-versus-token contact sheet.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 0f qualitative evidence | Report paths remain valid in the fitting worktree | Report ownership |
| Scientific review | No continuous appearance residual enters the token-only row | Representation semantics |

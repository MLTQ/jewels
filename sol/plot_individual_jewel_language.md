# `plot_individual_jewel_language.py`

## Purpose

Turns the passing Gate 0f report into the canonical representation-feasibility figure. Earlier
Gate 0 plots intentionally retain their failed verdicts; this plot shows the corrected original
proposal—one exact irregular centroid plus three active Jewel marks—without rewriting history.

## Components

### `load_metrics`

- **Does**: Validates the Gate 0f schema and extracts token render quality, spacetime-tilt
  retention, independent-fit language similarity, generation decisions, and center locking.

### `plot`

- **Does**: Displays every registered Gate 0f threshold beside its measured value and derives the
  title verdict from the report rather than a plotting argument.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Feasibility report | Active individual-language gate-v1 report | Input schema |
| Scientific review | Thresholds match `PROTOCOL_INDIVIDUAL_ACTIVE_V1.md` | Gate semantics |

# `test_block_token_jewel_speaker.py`

## Purpose

Checks the Gate 2 local expander's tensor contracts and verifies that generation preserves a
continuous irregular centroid output.

## Coverage

- Density and three active-token heads expose the registered shapes.
- A complete block program generates continuous centers and valid Jewel token rows.
- Incomplete block programs are rejected rather than silently broadcast.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a trainer | Stable loss and generation shapes | Head or program changes |

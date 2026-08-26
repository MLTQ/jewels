# `test_render_jewel_casting_language.py`

## Purpose

Protects deterministic, report-owned source selection for qualitative Gate-0 evidence.

## Components

### `CastingLanguageRenderTests`
- **Does**: Verifies lowest-seed selection and rejects incomplete validation coverage.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Qualitative renderer | Largest vocabulary contains rows for every registered validation source | Report records |

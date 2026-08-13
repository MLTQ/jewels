# `test_build_ltx_style_train.py`

## Purpose

Protects the four-field LTX cel-style adaptation manifest, especially its deliberately overlapping
training/reconstruction ownership and prompt provenance.

## Components

### `BuildLtxStyleTrainTests`

- **Does**: Verifies unique aliases point to the same physical video/fit, both splits contain the
  class, source overlap is explicit, prompt vectors are reused under a new digest, and mismatched or
  incomplete LTX fields are rejected. It also covers the exact serialized-file digest used by LTX
  generation receipts in addition to the canonical JSON digest used by prompt caches. Single-field
  selection must retain exactly one paired physical stem and reject missing classes.
- **Interacts with**: `build_ltx_style_train.py` and `prompt_embeddings.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Field loader | Train and validation aliases have unique source IDs but one shared video stem | Alias construction |
| Reporting | Reconstruction validation is never labelled unseen | Validation metadata |
| Prompt conditioning | Frozen embedding rows preserve their exact order and values | Cache rebinding |
| Memorization gate | Class selection yields one train/validation alias pair | Subset policy |

# `test_build_ltx_realizer_eval.py`

## Purpose

Protects the leakage boundary and prompt provenance of the UCF-train/LTX-validation realizer test.

## Components

### `BuildLtxRealizerEvalTests`

- **Does**: Confirms UCF training rows remain byte-equivalent, LTX replaces only validation,
  frozen prompt embeddings can be rebound under a new digest, and prompt/source-manifest
  mismatches are rejected.
- **Interacts with**: `build_ltx_realizer_eval.py`, `prompt_embeddings.py`, and the UCF prompt
  manifest contract.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Cross-domain render gate | Original train examples cannot be replaced or reordered | Split policy |
| Prompt controls | Same vectors receive new explicit LTX ownership and digest | Cache construction |

# `test_generate_ltx_corpus.py`

## Purpose

Protects the balanced LTX corpus plan, deterministic identities, and receipt-based resume contract
without importing LTX, PyTorch, or CUDA.

## Coverage

- Four classes produce three training and one evaluation sample apiece.
- Seeds and filenames are stable and unique across the 16-sample plan.
- Evaluation-role selection preserves one held-out prompt and its original seed per class.
- Source and generation prompts remain separately recoverable.
- Only an MP4 plus a configuration-matching successful receipt counts as complete.
- The corpus manifest exposes receipt metrics and rejects unsupported source schemas.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `generate_ltx_corpus.py` | Tests use only standard-library temporary files | External imports |
| Resume service | A changed prompt, seed, geometry, or memory policy forces regeneration | Match fields |

# `test_ltx_scaffold.py`

## Purpose

Protects the reproducible external LTX scaffold command without importing or installing LTX in the
Jewels environment.

## Coverage

- Prompts containing quotes and shell punctuation remain one subprocess argument.
- Width/height obey the official 64-pixel multiple and frame count obeys `8*K+1`.
- Missing checkpoint, upscaler, text-encoder, or virtual-environment assets fail before launch.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `ltx_scaffold.py` | Pure command construction is locally testable | CLI argument order/names |
| CI | Tests require only the Python standard library | Importing LTX or CUDA packages |

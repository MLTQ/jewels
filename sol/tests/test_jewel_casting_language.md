# `test_jewel_casting_language.py`

## Purpose

Protects Gate 0 from becoming a lossy grid tokenizer by accident. The tests require lossless target
coverage, continuous-center controls, bundle-level casting, deterministic histogram comparison,
and a visible distinction between motif-only and residual-complete decoding.

## Components

### `JewelCastingLanguageTests`
- **Does**: Fits a tiny motif vocabulary over synthetic irregular fields and exercises the complete
  encode/decode program contract.
- **Does**: Confirms that cell-center quantization is isolated as a negative control.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate-0 implementation | Every source Jewel survives serialization and residual-complete decode | Count handling |
| Future generator | One motif cast may own multiple Jewels | Bundle semantics |

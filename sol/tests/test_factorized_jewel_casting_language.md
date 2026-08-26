# `test_factorized_jewel_casting_language.py`

## Purpose

Protects the role partition, exact round trip, no-drop invariant, discrete-decision accounting, and
portable codebook artifact of the factorized casting phrase.

## Components

### `FactorizedCastingLanguageTests`
- **Does**: Fits tiny synthetic role vocabularies and validates composition and artifact reload
  without a renderer/GPU.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Factorized language | Role dimensions exhaust canonical 22D features exactly once | Factor partition |
| Gate 0b | Four role decisions per cast and exact full residual | Program accounting |

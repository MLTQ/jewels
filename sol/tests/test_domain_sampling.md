# `test_domain_sampling.py`

## Purpose

Protects the mixed-domain experiment from silently reverting to corpus-size-weighted training.

## Components

### `DomainSamplingTests`

- **Does**: verifies exact domain alternation at batch one, one example per domain at batch two, and
  validation of empty or non-positive requests
- **Interacts with**: `domain_sampling.py`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Joint tokenizer experiment | Two domains receive equal step counts despite a 86:1 window ratio | Selection policy |
| Reproducibility | Domain order is deterministic and within-domain draws use the passed generator | RNG ownership |

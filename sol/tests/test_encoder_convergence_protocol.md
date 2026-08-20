# `test_encoder_convergence_protocol.py`

## Purpose

Protects the scientific invariants behind the corrected encoder data curve.

## Components

### `EncoderConvergenceProtocolTests`
- Accepts exact nested training prefixes with frozen ordered validation identities.
- Rejects equal-size membership that changes prefix order.
- Verifies the three-seed Student-t interval is centered on the sample mean.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Convergence runner | Bad nesting or validation drift fails before GPU work | Protocol validation |
| Feasibility report | Uncertainty derives from visible seed observations | Confidence calculation |

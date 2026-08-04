# `test_token_grid.py`

## Purpose

Protects the dense tokenizer gate: 45k jewels must fit losslessly at the declared capacity, overflow
must fail loudly, occupancy must survive pooling, and the proposed autoencoder loss must train.

## Components

### `TokenGridTests`
- **Does**: Covers dense/compact packing, capacity, overflow, statistics, and a gradient smoke test.
- **Interacts with**: `token_grid.py`, `autoencoder.py`, and `synthetic.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Dense corpus training | No target jewel is silently discarded | Packing/capacity policy |
| Autoencoder development | Feature, existence, and count losses remain differentiable | Loss outputs |

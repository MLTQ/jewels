# `test_hierarchical_jewel_decoder.py`

## Purpose

Protects phrase/token alignment, terminal-pair padding, sampled/full construction parity, masked
training, output shape, and exact target decoding for the learned product-code decoder.

## Components

### `HierarchicalJewelDecoderTests`
- **Does**: Fits tiny source-owned vocabularies and exercises the entire data-to-model-to-feature
  contract without a renderer or GPU.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Decoder gate | Target residual is a loss target, never a forward input | Batch/model boundary |
| Hierarchical phrase | Padding only occupies missing second surface/gradient roles | Token alignment |

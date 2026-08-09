"""Restore tokenizer architectures from explicit checkpoint metadata."""

from __future__ import annotations

import torch.nn as nn

from sol.grouped_sparse_autoencoder import GroupedSparseJewelAutoencoder
from sol.sparse_autoencoder import SparseJewelAutoencoder
from sol.token_grid import GridSpec


def build_tokenizer(meta: dict, spec: GridSpec) -> nn.Module:
    """Construct the checkpoint-declared tokenizer without loading its weights."""
    architecture = meta.get("architecture")
    model_args = dict(meta["model_args"])
    model_args["spec"] = spec
    if architecture == "sparse_variable_count_v1":
        return SparseJewelAutoencoder(**model_args)
    if architecture == "grouped_sparse_tokens_v1":
        return GroupedSparseJewelAutoencoder(**model_args)
    raise ValueError(f"unsupported tokenizer architecture: {architecture}")

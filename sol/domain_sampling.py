"""Deterministic domain-balanced sampling for mixed fitted corpora."""

from __future__ import annotations

import torch


def sample_domain_balanced_indices(
    domain_ids: list[str],
    batch_size: int,
    step: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Choose domains round-robin and examples uniformly within each domain."""
    if not domain_ids:
        raise ValueError("domain-balanced sampling requires examples")
    if batch_size <= 0 or step <= 0:
        raise ValueError("batch size and step must be positive")
    by_domain: dict[str, list[int]] = {}
    for index, domain_id in enumerate(domain_ids):
        by_domain.setdefault(domain_id, []).append(index)
    domains = sorted(by_domain)
    start = ((step - 1) * batch_size) % len(domains)
    selected = []
    for offset in range(batch_size):
        domain = domains[(start + offset) % len(domains)]
        choices = by_domain[domain]
        pick = int(torch.randint(0, len(choices), (), generator=generator))
        selected.append(choices[pick])
    return torch.tensor(selected, dtype=torch.long)

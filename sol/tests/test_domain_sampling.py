"""Tests for deterministic domain-balanced training selection."""

from __future__ import annotations

import unittest

import torch

from sol.domain_sampling import sample_domain_balanced_indices


class DomainSamplingTests(unittest.TestCase):
    def test_batch_one_alternates_domains_despite_imbalance(self) -> None:
        domains = ["avenue", "avenue", "avenue", "ucf"]
        generator = torch.Generator().manual_seed(4)
        sampled_domains = []
        for step in range(1, 9):
            index = int(sample_domain_balanced_indices(domains, 1, step, generator)[0])
            sampled_domains.append(domains[index])
        self.assertEqual(sampled_domains, ["avenue", "ucf"] * 4)

    def test_batch_two_contains_each_domain(self) -> None:
        domains = ["avenue", "avenue", "avenue", "ucf"]
        generator = torch.Generator().manual_seed(7)
        for step in range(1, 5):
            indices = sample_domain_balanced_indices(domains, 2, step, generator)
            self.assertEqual({domains[int(index)] for index in indices}, {"avenue", "ucf"})

    def test_rejects_invalid_requests(self) -> None:
        generator = torch.Generator().manual_seed(0)
        with self.assertRaises(ValueError):
            sample_domain_balanced_indices([], 1, 1, generator)
        with self.assertRaises(ValueError):
            sample_domain_balanced_indices(["avenue"], 0, 1, generator)


if __name__ == "__main__":
    unittest.main()

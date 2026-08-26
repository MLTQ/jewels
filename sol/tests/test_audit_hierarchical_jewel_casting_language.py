"""Tests for hierarchical casting-language composition and statistics."""

from __future__ import annotations

import unittest

import torch

from sol.audit_hierarchical_jewel_casting_language import (
    compose_hierarchical_features,
    hierarchical_decisions,
    pairwise_hierarchical_similarity,
)
from sol.factorized_jewel_casting_language import (
    decode_factorized_program,
    encode_factorized_program,
    fit_factorized_codebook,
)
from sol.synthetic import random_jewels
from sol.token_grid import GridSpec


class HierarchicalCastingAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = GridSpec((2, 2, 2), slots_per_cell=1)
        self.fields = [random_jewels(96, seed=seed) for seed in range(4)]
        self.pair_codebook, _ = fit_factorized_codebook(
            self.fields[:2], spec=self.spec, bundle_size=2,
            vocabulary_size=8, iterations=2, assignment_chunk=16,
        )
        self.individual_codebook, _ = fit_factorized_codebook(
            self.fields[:2], spec=self.spec, bundle_size=1,
            vocabulary_size=8, iterations=2, assignment_chunk=16,
        )

    def _programs(self, field: torch.Tensor):
        return (
            encode_factorized_program(field, self.pair_codebook),
            encode_factorized_program(field, self.individual_codebook),
        )

    def test_full_hierarchy_round_trip_is_exact(self) -> None:
        pair, individual = self._programs(self.fields[2])
        decoded = compose_hierarchical_features(
            decode_factorized_program(pair, self.pair_codebook),
            decode_factorized_program(individual, self.individual_codebook),
        )
        for dimension in range(22):
            torch.testing.assert_close(
                decoded[:, dimension].sort().values,
                self.fields[2][:, dimension].sort().values,
                rtol=1e-5, atol=1e-5,
            )
        self.assertEqual(
            hierarchical_decisions(pair, individual), pair.casts * 2 + len(self.fields[2]) * 2
        )

    def test_repeated_program_has_positive_similarity_margin(self) -> None:
        pair, individual = self._programs(self.fields[2])
        other_pair, other_individual = self._programs(self.fields[3])
        report = pairwise_hierarchical_similarity(
            [
                ("same", pair, individual),
                ("same", pair, individual),
                ("different", other_pair, other_individual),
            ],
            n_cells=self.spec.n_cells,
            vocabulary_size=8,
        )
        self.assertGreater(report["summary"]["margin"], 0)


if __name__ == "__main__":
    unittest.main()

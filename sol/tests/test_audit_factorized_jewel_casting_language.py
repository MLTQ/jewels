"""Tests for factorized casting-language audit statistics."""

from __future__ import annotations

import unittest

from sol.audit_factorized_jewel_casting_language import (
    factorized_residual_metrics,
    pairwise_factorized_similarity,
)
from sol.factorized_jewel_casting_language import (
    encode_factorized_program,
    fit_factorized_codebook,
)
from sol.synthetic import random_jewels
from sol.token_grid import GridSpec


class FactorizedCastingAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = GridSpec((2, 2, 2), slots_per_cell=1)
        self.fields = [random_jewels(96, seed=seed) for seed in range(4)]
        self.codebook, _ = fit_factorized_codebook(
            self.fields[:2], spec=self.spec, bundle_size=4,
            vocabulary_size=8, iterations=2, assignment_chunk=16,
        )

    def test_metrics_audit_no_drop_and_role_energy(self) -> None:
        program = encode_factorized_program(self.fields[2], self.codebook)
        report = factorized_residual_metrics(program, self.codebook)
        self.assertEqual(report["source_jewels"], len(self.fields[2]))
        self.assertEqual(report["serialized_jewels"], len(self.fields[2]))
        self.assertEqual(set(report["factors"]), {"layout", "covariance", "surface", "gradient"})

    def test_pairwise_exact_repeat_has_positive_composite_margin(self) -> None:
        repeated = encode_factorized_program(self.fields[2], self.codebook)
        different = encode_factorized_program(self.fields[3], self.codebook)
        report = pairwise_factorized_similarity(
            [("same", repeated), ("same", repeated), ("different", different)],
            n_cells=self.spec.n_cells,
            vocabulary_size=self.codebook.vocabulary_size,
        )
        self.assertGreater(
            report["summary"]["composite_cell_conditional_cosine"]["margin"], 0
        )


if __name__ == "__main__":
    unittest.main()

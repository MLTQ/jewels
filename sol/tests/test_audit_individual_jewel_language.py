"""Tests for active individual-Jewel language similarity."""

from __future__ import annotations

import unittest

from sol.audit_individual_jewel_language import (
    ACTIVE_FACTORS,
    pairwise_active_similarity,
)
from sol.factorized_jewel_casting_language import (
    encode_factorized_program,
    fit_factorized_codebook,
)
from sol.synthetic import random_jewels
from sol.token_grid import GridSpec


class ActiveIndividualLanguageTests(unittest.TestCase):
    def test_constant_layout_is_not_an_active_factor(self) -> None:
        self.assertEqual(ACTIVE_FACTORS, ("covariance", "surface", "gradient"))

    def test_repeated_field_has_positive_active_margin(self) -> None:
        spec = GridSpec((2, 2, 2), slots_per_cell=1)
        fields = [random_jewels(96, seed=seed) for seed in range(4)]
        codebook, _ = fit_factorized_codebook(
            fields[:2], spec=spec, bundle_size=1,
            vocabulary_size=8, iterations=2, assignment_chunk=16,
        )
        repeated = encode_factorized_program(fields[2], codebook)
        different = encode_factorized_program(fields[3], codebook)
        report = pairwise_active_similarity(
            [("same", repeated), ("same", repeated), ("different", different)],
            n_cells=spec.n_cells,
            vocabulary_size=codebook.vocabulary_size,
        )
        self.assertGreater(report["summary"]["margin"], 0)


if __name__ == "__main__":
    unittest.main()

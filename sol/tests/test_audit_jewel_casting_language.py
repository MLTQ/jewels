"""Tests for Jewel casting language audit statistics."""

from __future__ import annotations

import unittest

from sol.audit_jewel_casting_language import (
    center_irregularity,
    pairwise_language_similarity,
    residual_metrics,
)
from sol.jewel_casting_language import (
    encode_program,
    fit_motif_codebook,
    quantize_centers_to_cells,
)
from sol.synthetic import random_jewels
from sol.token_grid import GridSpec


class JewelCastingAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = GridSpec((2, 2, 2), slots_per_cell=1)
        self.fields = [random_jewels(96, seed=seed) for seed in range(4)]
        self.codebook, _ = fit_motif_codebook(
            self.fields[:2], spec=self.spec, bundle_size=4,
            vocabulary_size=8, iterations=3, assignment_chunk=16,
        )

    def test_residual_metrics_report_bundle_decision_count(self) -> None:
        program = encode_program(self.fields[2], self.codebook)
        report = residual_metrics(program, self.codebook)
        self.assertLess(report["casts"], len(self.fields[2]))
        self.assertGreater(report["jewels_per_cast"], 1)
        self.assertEqual(report["source_jewels"], len(self.fields[2]))
        self.assertEqual(report["serialized_jewels"], len(self.fields[2]))

    def test_quantized_control_is_completely_cell_locked(self) -> None:
        quantized = quantize_centers_to_cells(self.fields[2], self.spec)
        report = center_irregularity(quantized, self.spec)
        self.assertEqual(report["cell_center_lock_fraction"], 1.0)

    def test_pairwise_report_separates_same_and_different_sources(self) -> None:
        repeated = encode_program(self.fields[2], self.codebook)
        different = encode_program(self.fields[3], self.codebook)
        report = pairwise_language_similarity(
            [("same", repeated), ("same", repeated), ("different", different)],
            n_cells=self.spec.n_cells,
            vocabulary_size=self.codebook.vocabulary_size,
        )
        self.assertGreater(
            report["summary"]["raw_motif_cosine"]["margin"], 0
        )


if __name__ == "__main__":
    unittest.main()

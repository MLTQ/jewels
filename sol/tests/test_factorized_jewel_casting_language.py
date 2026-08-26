"""Tests for compositional Jewel casting programs."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import torch

from sol.factorized_jewel_casting_language import (
    FACTOR_DIMENSIONS,
    decode_factorized_program,
    encode_factorized_program,
    factor_histograms,
    fit_factorized_codebook,
    load_factorized_codebook,
)
from sol.audit_factorized_jewel_casting_language import _save_codebook
from sol.synthetic import random_jewels
from sol.token_grid import GridSpec


class FactorizedCastingLanguageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = GridSpec((2, 2, 2), slots_per_cell=1)
        self.fields = [random_jewels(96, seed=seed) for seed in range(3)]
        self.codebook, _ = fit_factorized_codebook(
            self.fields[:2], spec=self.spec, bundle_size=4,
            vocabulary_size=8, iterations=2, assignment_chunk=16,
        )

    def test_factor_partition_is_exhaustive_and_disjoint(self) -> None:
        dimensions = [value for group in FACTOR_DIMENSIONS.values() for value in group]
        self.assertEqual(sorted(dimensions), list(range(22)))

    def test_full_residual_round_trip_is_exact_as_a_set(self) -> None:
        program = encode_factorized_program(self.fields[2], self.codebook)
        decoded = decode_factorized_program(program, self.codebook)
        self.assertEqual(len(decoded), len(self.fields[2]))
        for dimension in range(22):
            torch.testing.assert_close(
                decoded[:, dimension].sort().values,
                self.fields[2][:, dimension].sort().values,
                rtol=1e-5, atol=1e-5,
            )

    def test_program_emits_one_decision_per_factor_per_cast(self) -> None:
        program = encode_factorized_program(self.fields[2], self.codebook)
        self.assertEqual(program.discrete_decisions, program.casts * 4)
        self.assertEqual(int(program.counts.sum()), len(self.fields[2]))
        histograms = factor_histograms(
            program,
            n_cells=self.spec.n_cells,
            vocabulary_size=self.codebook.vocabulary_size,
        )
        self.assertEqual(set(histograms), set(FACTOR_DIMENSIONS))

    def test_saved_codebook_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "codebook.pt"
            _save_codebook(self.codebook, path)
            restored = load_factorized_codebook(path)
        self.assertEqual(restored.bundle_size, self.codebook.bundle_size)
        self.assertEqual(
            [factor.name for factor in restored.factors],
            [factor.name for factor in self.codebook.factors],
        )


if __name__ == "__main__":
    unittest.main()

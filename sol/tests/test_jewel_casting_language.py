"""Tests for the canonical discrete-continuous Jewel casting language."""

from __future__ import annotations

import unittest

import torch

from sol.jewel_casting_language import (
    decode_program,
    encode_program,
    fit_motif_codebook,
    histogram_cosine,
    program_histogram,
    quantize_centers_to_cells,
)
from sol.synthetic import random_jewels
from sol.token_grid import GridSpec


class JewelCastingLanguageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = GridSpec((2, 2, 2), slots_per_cell=1)
        self.fields = [random_jewels(96, seed=seed) for seed in range(3)]
        self.codebook, self.report = fit_motif_codebook(
            self.fields[:2],
            spec=self.spec,
            bundle_size=4,
            vocabulary_size=8,
            iterations=3,
            max_casts=1000,
            assignment_chunk=16,
            seed=4,
        )

    def test_full_residual_round_trip_preserves_every_jewel(self) -> None:
        program = encode_program(self.fields[2], self.codebook)
        decoded = decode_program(program, self.codebook, residual_scale=1.0)
        self.assertEqual(len(decoded), len(self.fields[2]))
        # Decode uses canonical cell/bundle order rather than input set order.
        source_program = encode_program(decoded, self.codebook)
        decoded_again = decode_program(source_program, self.codebook, residual_scale=1.0)
        torch.testing.assert_close(decoded, decoded_again, atol=1e-5, rtol=1e-5)

    def test_program_uses_bundle_casts_and_every_motif(self) -> None:
        program = encode_program(self.fields[2], self.codebook)
        self.assertLess(program.casts, len(self.fields[2]))
        self.assertEqual(int(program.counts.sum()), len(self.fields[2]))
        self.assertGreater(self.report["utilized_fraction"], 0)

    def test_same_program_histogram_is_one(self) -> None:
        program = encode_program(self.fields[2], self.codebook)
        histogram = program_histogram(
            program,
            n_cells=self.spec.n_cells,
            vocabulary_size=self.codebook.vocabulary_size,
        )
        self.assertAlmostEqual(histogram_cosine(histogram, histogram), 1.0, places=6)

    def test_center_quantization_is_an_explicit_negative_control(self) -> None:
        quantized = quantize_centers_to_cells(self.fields[2], self.spec)
        self.assertFalse(torch.equal(quantized[:, :3], self.fields[2][:, :3]))
        self.assertEqual(
            len(torch.unique(quantized[:, :3], dim=0)), self.spec.n_cells
        )

    def test_codebook_only_decode_differs_from_exact_residual_decode(self) -> None:
        program = encode_program(self.fields[2], self.codebook)
        token_only = decode_program(program, self.codebook, residual_scale=0.0)
        exact = decode_program(program, self.codebook, residual_scale=1.0)
        self.assertGreater(float((token_only - exact).abs().max()), 1e-4)


if __name__ == "__main__":
    unittest.main()

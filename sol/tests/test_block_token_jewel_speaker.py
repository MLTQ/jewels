"""Tests for expanding discrete block tokens into irregular Jewel casts."""

from __future__ import annotations

import unittest

import torch

from sol.block_token_jewel_speaker import BlockTokenJewelSpeaker


class BlockTokenJewelSpeakerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = BlockTokenJewelSpeaker(
            block_vocabulary_size=11,
            jewel_vocabulary_size=7,
            block_shape=(4, 4, 2),
            hidden_dim=16,
            depth=1,
        )
        self.program = torch.arange(32).remainder(11).long()

    def test_heads_have_expected_shapes(self) -> None:
        centers = torch.rand(13, 3) * 2 - 1
        block_tokens = self.model.program_tokens(self.program, centers)
        self.assertEqual(self.model.token_logits(block_tokens, centers).shape, (13, 3, 7))
        self.assertEqual(self.model.intensity_logits(block_tokens, centers).shape, (13,))

    def test_sampling_returns_continuous_centers_and_active_tokens(self) -> None:
        generator = torch.Generator().manual_seed(8)
        centers = self.model.sample_centers(
            self.program, 23, generator=generator, proposal_multiplier=2
        )
        tokens = self.model.sample_tokens(
            self.program, centers, generator=generator, top_k=3
        )
        self.assertEqual(centers.shape, (23, 3))
        self.assertEqual(tokens.shape, (23, 3))
        self.assertGreater(float(centers.std()), 0.1)

    def test_program_length_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "one token"):
            self.model.program_tokens(torch.zeros(3, dtype=torch.long), torch.rand(2, 3))


if __name__ == "__main__":
    unittest.main()

"""Fixed block-PCA codec tests."""

from __future__ import annotations

import unittest

import torch

from sol.block_codec import BlockPCACodec, fit_block_pca


class BlockCodecTests(unittest.TestCase):
    def test_full_basis_roundtrip_is_exact(self) -> None:
        generator = torch.Generator().manual_seed(51)
        latents = torch.randn(6, 64, 3, generator=generator)
        codec = fit_block_pca(
            latents,
            (4, 4, 4),
            block_size=2,
            code_dim=24,
            max_blocks=48,
        )
        reconstructed = codec.decode(codec.encode(latents))
        torch.testing.assert_close(reconstructed, latents, atol=2e-5, rtol=2e-5)

    def test_state_roundtrip_preserves_codes(self) -> None:
        latents = torch.randn(4, 64, 2, generator=torch.Generator().manual_seed(52))
        codec = fit_block_pca(
            latents, (4, 4, 4), block_size=2, code_dim=8, max_blocks=32
        )
        restored = BlockPCACodec.from_state_dict(codec.state_dict())
        self.assertEqual(restored.coarse_shape, (2, 2, 2))
        torch.testing.assert_close(restored.encode(latents), codec.encode(latents))


if __name__ == "__main__":
    unittest.main()

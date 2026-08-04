"""Dense latent hierarchy diagnostic tests."""

from __future__ import annotations

import unittest

import torch

from sol.latent_hierarchy import block_vectors, hierarchy_report, reshape_latents


class LatentHierarchyTests(unittest.TestCase):
    def test_flat_raster_roundtrip_and_block_order(self) -> None:
        latents = torch.arange(2 * 64 * 3).reshape(2, 64, 3).float()
        volume = reshape_latents(latents, (4, 4, 4))
        torch.testing.assert_close(volume.reshape_as(latents), latents)
        blocks = block_vectors(volume, 2)
        self.assertEqual(blocks.shape, (16, 24))

    def test_constant_blocks_have_zero_repeat_mean_error(self) -> None:
        volume = torch.zeros(3, 4, 4, 4, 2)
        for u in range(2):
            for v in range(2):
                for t in range(2):
                    volume[:, u * 2 : u * 2 + 2, v * 2 : v * 2 + 2, t * 2 : t * 2 + 2] = (
                        u * 4 + v * 2 + t
                    )
        report = hierarchy_report(
            volume.reshape(3, 64, 2),
            (4, 4, 4),
            block_sizes=(2,),
            pca_dimensions=(2, 4),
            max_pca_blocks=24,
        )
        self.assertAlmostEqual(report["pooling"]["2"]["repeat_mean_mse"], 0.0)
        explained = report["pca"]["explained_variance"]
        self.assertLessEqual(explained["2"], explained["4"])


if __name__ == "__main__":
    unittest.main()

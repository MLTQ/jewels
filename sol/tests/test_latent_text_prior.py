"""Tests for the text-conditioned latent flow."""

from __future__ import annotations

import unittest

import torch

from sol.latent_text_prior import LatentStandardizer, LatentTextPrior
from sol.train_latent_text_prior import pack, unpack


def _model(n_cells: int = 12, cell_dim: int = 8, slots: int = 4) -> LatentTextPrior:
    torch.manual_seed(0)
    return LatentTextPrior(
        n_cells=n_cells,
        cell_dim=cell_dim,
        seed_dim=slots * 3,
        text_dim=16,
        model_dim=32,
        depth=2,
        heads=4,
    )


class LatentTextPriorTests(unittest.TestCase):
    def test_pack_unpack_round_trips(self) -> None:
        cells = torch.randn(3, 12, 8)
        seed = torch.rand(3, 12, 4, 3)
        packed = pack(cells, seed)
        self.assertEqual(packed.shape, (3, 12, 8 + 12))
        back_cells, back_seed = unpack(packed, 8, 4)
        self.assertTrue(torch.equal(back_cells, cells))
        self.assertTrue(torch.equal(back_seed, seed))

    def test_standardizer_round_trips(self) -> None:
        values = torch.randn(5, 12, 20) * 3 + 1
        standardizer = LatentStandardizer.fit(values)
        normalized = standardizer.normalize(values)
        self.assertLess(float(normalized.mean().abs()), 0.2)
        self.assertTrue(
            torch.allclose(standardizer.denormalize(normalized), values, atol=1e-4)
        )

    def test_zero_init_output_is_zero_velocity(self) -> None:
        model = _model()
        noisy = torch.randn(2, 12, 20)
        velocity = model(
            noisy, torch.rand(2), torch.randn(2, 6, 16), torch.ones(2, 6, dtype=torch.long)
        )
        self.assertTrue(torch.equal(velocity, torch.zeros_like(velocity)))

    def test_text_changes_output_once_trained(self) -> None:
        model = _model()
        with torch.no_grad():
            model.output_projection.weight.normal_(0, 0.05)
            for block in model.blocks:
                block.modulation[-1].weight.normal_(0, 0.05)
                block.modulation[-1].bias.normal_(0.5, 0.05)
        noisy = torch.randn(2, 12, 20)
        flow_time = torch.rand(2)
        mask = torch.ones(2, 6, dtype=torch.long)
        torch.manual_seed(1)
        first = model(noisy, flow_time, torch.randn(2, 6, 16), mask)
        torch.manual_seed(1)
        second = model(noisy, flow_time, torch.randn(2, 6, 16) * 3, mask)
        self.assertGreater(float((first - second).abs().max()), 1e-6)

    def test_sampling_returns_latent_shaped_output(self) -> None:
        model = _model()
        out = model.sample(
            torch.randn(2, 6, 16),
            torch.ones(2, 6, dtype=torch.long),
            steps=3,
            generator=torch.Generator().manual_seed(0),
        )
        self.assertEqual(out.shape, (2, 12, 20))

    def test_null_conditioning_path_runs(self) -> None:
        model = _model()
        velocity = model(torch.randn(2, 12, 20), torch.rand(2), None, None)
        self.assertEqual(velocity.shape, (2, 12, 20))


if __name__ == "__main__":
    unittest.main()

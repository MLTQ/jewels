"""Shape, conditioning, and gradient tests for the raster latent prior."""

from __future__ import annotations

import unittest

import torch

from sol.latent_prior import (
    RasterFlowPrior,
    flow_matching_loss,
    flow_matching_objective,
    masked_flow_matching_objective,
    sample_flow,
)
from sol.prior_evaluation import energy_distance


class LatentPriorTests(unittest.TestCase):
    def test_flow_loss_backpropagates(self) -> None:
        model = RasterFlowPrior(
            n_cells=8,
            latent_dim=6,
            model_dim=16,
            depth=2,
            heads=4,
            text_dim=10,
        )
        latents = torch.randn(3, 8, 6)
        text = torch.randn(3, 10)
        loss = flow_matching_loss(model, latents, text, condition_dropout=0.25)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_prior_matches_inpainting_callable_contract(self) -> None:
        model = RasterFlowPrior(
            n_cells=4,
            latent_dim=3,
            model_dim=12,
            depth=1,
            heads=3,
            text_dim=5,
        )
        latents = torch.randn(2, 4, 3)
        time = torch.rand(2)
        text = torch.randn(2, 5)
        self.assertEqual(model(latents, time, text).shape, latents.shape)
        self.assertEqual(model(latents, time, None).shape, latents.shape)

    def test_explicit_flow_path_and_sampler(self) -> None:
        model = RasterFlowPrior(
            n_cells=4,
            latent_dim=3,
            model_dim=12,
            depth=1,
            heads=3,
            text_dim=5,
        )
        target = torch.randn(2, 4, 3)
        condition = torch.randn(2, 5)
        noise = torch.randn_like(target)
        time = torch.tensor([0.25, 0.75])
        loss = flow_matching_objective(
            model, target, condition, noise=noise, time=time
        )
        self.assertTrue(torch.isfinite(loss))
        generated = sample_flow(
            model,
            condition,
            batch=2,
            n_cells=4,
            latent_dim=3,
            device="cpu",
            steps=3,
            generator=torch.Generator().manual_seed(4),
        )
        self.assertEqual(generated.shape, target.shape)

    def test_energy_distance_prefers_matching_distribution_to_collapse(self) -> None:
        target = torch.tensor([[-1.0], [1.0]])
        matching = target.clone()
        collapsed = torch.zeros_like(target)
        self.assertAlmostEqual(energy_distance(target, matching), 0.0)
        self.assertLess(
            energy_distance(target, matching), energy_distance(collapsed, target)
        )

    def test_masked_flow_path_backpropagates_only_scored_region(self) -> None:
        model = RasterFlowPrior(
            n_cells=4,
            latent_dim=3,
            model_dim=12,
            depth=1,
            heads=3,
            text_dim=5,
        )
        target = torch.randn(2, 4, 3)
        condition = torch.randn(2, 5)
        dirty = torch.tensor([[True, False, False, True], [False, True, True, False]])
        loss = masked_flow_matching_objective(
            model,
            target,
            condition,
            dirty,
            noise=torch.randn_like(target),
            time=torch.tensor([0.2, 0.8]),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_masked_flow_rejects_empty_region(self) -> None:
        model = RasterFlowPrior(
            n_cells=4,
            latent_dim=3,
            model_dim=12,
            depth=1,
            heads=3,
            text_dim=5,
        )
        with self.assertRaises(ValueError):
            masked_flow_matching_objective(
                model,
                torch.randn(1, 4, 3),
                torch.randn(1, 5),
                torch.zeros(4, dtype=torch.bool),
                noise=torch.randn(1, 4, 3),
                time=torch.tensor([0.5]),
            )


if __name__ == "__main__":
    unittest.main()

"""Axial hierarchical flow shape, gradient, and sampling tests."""

from __future__ import annotations

import unittest

import torch

from sol.axial_prior import AxialFlowPrior
from sol.evaluate_latent_prior import _restore_prior
from sol.latent_prior import flow_matching_loss, sample_flow


class AxialPriorTests(unittest.TestCase):
    def _model(self) -> AxialFlowPrior:
        return AxialFlowPrior(
            grid_shape=(2, 2, 2),
            latent_dim=6,
            model_dim=16,
            depth=3,
            heads=4,
            text_dim=10,
        )

    def test_axial_flow_backpropagates_and_matches_shape(self) -> None:
        model = self._model()
        latents = torch.randn(3, 8, 6)
        condition = torch.randn(3, 10)
        loss = flow_matching_loss(model, latents, condition)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        self.assertEqual(model(latents, torch.rand(3), None).shape, latents.shape)

    def test_generic_sampler_accepts_axial_prior(self) -> None:
        model = self._model()
        generated = sample_flow(
            model,
            torch.randn(2, 10),
            batch=2,
            n_cells=8,
            latent_dim=6,
            device="cpu",
            steps=3,
            generator=torch.Generator().manual_seed(61),
        )
        self.assertEqual(generated.shape, (2, 8, 6))

    def test_evaluator_restores_axial_checkpoint(self) -> None:
        model = self._model()
        checkpoint = {
            "model": model.state_dict(),
            "ema": model.state_dict(),
            "meta": {
                "architecture": "axial_flow_v1",
                "model_args": {
                    "grid_shape": (2, 2, 2),
                    "latent_dim": 6,
                    "model_dim": 16,
                    "depth": 3,
                    "heads": 4,
                    "text_dim": 10,
                },
            },
        }
        self.assertIsInstance(_restore_prior(checkpoint), AxialFlowPrior)

    def test_mask_conditioned_prior_accepts_explicit_edit_mask(self) -> None:
        model = AxialFlowPrior(
            grid_shape=(2, 2, 2),
            latent_dim=6,
            model_dim=16,
            depth=3,
            heads=4,
            text_dim=10,
            mask_conditioning=True,
        )
        latents = torch.randn(2, 8, 6)
        dirty = torch.zeros(2, 8, dtype=torch.bool)
        dirty[:, :2] = True
        output = model(latents, torch.rand(2), torch.randn(2, 10), edit_mask=dirty)
        self.assertEqual(output.shape, latents.shape)
        with self.assertRaises(ValueError):
            model(latents, torch.rand(2), None, edit_mask=dirty[:, :-1])


if __name__ == "__main__":
    unittest.main()

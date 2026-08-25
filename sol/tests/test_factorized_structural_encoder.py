"""Tests for factorized irregular geometry and appearance."""

from __future__ import annotations

import unittest

import torch

from sol.factorized_structural_encoder import (
    FactorizedStructuralJewelEncoder,
    native_neighborhood_derivatives,
    sample_native_neighborhood,
)
from sol.structural_encoder import StructuralJewelEncoder
from sol.token_grid import GridSpec


class FactorizedStructuralEncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = GridSpec((2, 2, 2), 1)
        self.video = torch.rand(5, 16, 16, 3)

    def test_prediction_and_canonical_shapes(self) -> None:
        model = FactorizedStructuralJewelEncoder(
            grid_spec=self.spec, slots_per_cell=3, model_dim=8,
            appearance_dim=8, appearance_hidden=16,
        )
        prediction = model(self.video)
        self.assertEqual(prediction["centers"].shape, (24, 3))
        self.assertEqual(prediction["color_grads"].shape, (24, 3, 3))
        self.assertEqual(prediction["appearance_residual"].shape, (24, 12))
        self.assertEqual(prediction["appearance_adapter_residual"].shape, (24, 12))
        self.assertEqual(float(prediction["appearance_residual"].abs().sum()), 0.0)
        self.assertEqual(model.canonical_features(prediction).shape, (24, 22))

    def test_appearance_loss_does_not_reach_geometry(self) -> None:
        model = FactorizedStructuralJewelEncoder(
            grid_spec=self.spec, slots_per_cell=2, model_dim=8,
            appearance_dim=8, appearance_hidden=16,
        )
        prediction = model(self.video)
        prediction["colors"].square().mean().backward()
        self.assertTrue(all(
            parameter.grad is None for parameter in model.geometry_trunk.parameters()
        ))
        self.assertIsNone(model.geometry_head.weight.grad)
        self.assertIsNotNone(model.appearance_head[-1].weight.grad)

    def test_v2_transplant_preserves_geometry_exactly(self) -> None:
        torch.manual_seed(7)
        source = StructuralJewelEncoder(
            grid_spec=self.spec, slots_per_cell=3, model_dim=8,
            max_offset_cells=4.0, seed_video_colors=True,
        ).eval()
        target = FactorizedStructuralJewelEncoder(
            grid_spec=self.spec, slots_per_cell=3, model_dim=8,
            max_offset_cells=4.0, appearance_dim=8, appearance_hidden=16,
        ).eval()
        target.load_v2_geometry(source.state_dict())
        source_prediction = source(self.video)
        target_prediction = target(self.video)
        for key in ("centers", "log_scale", "quaternion", "logit_w"):
            torch.testing.assert_close(target_prediction[key], source_prediction[key])

    def test_freeze_geometry_leaves_appearance_trainable(self) -> None:
        model = FactorizedStructuralJewelEncoder(
            grid_spec=self.spec, slots_per_cell=2, model_dim=8,
            appearance_dim=8, appearance_hidden=16,
        )
        model.freeze_geometry()
        self.assertTrue(all(
            not parameter.requires_grad
            for parameter in (*model.geometry_trunk.parameters(), *model.geometry_head.parameters())
        ))
        self.assertTrue(all(
            parameter.requires_grad for parameter in model.appearance_head.parameters()
        ))

    def test_residual_expansion_preserves_bounded_checkpoint_exactly(self) -> None:
        torch.manual_seed(11)
        bounded = FactorizedStructuralJewelEncoder(
            grid_spec=self.spec, slots_per_cell=2, model_dim=8,
            appearance_dim=8, appearance_hidden=16,
        ).eval()
        expanded = FactorizedStructuralJewelEncoder(
            grid_spec=self.spec, slots_per_cell=2, model_dim=8,
            appearance_dim=8, appearance_hidden=16,
            appearance_contract="residual",
        ).eval()
        expanded.load_bounded_appearance_expansion(bounded.state_dict())
        bounded_prediction = bounded(self.video)
        expanded_prediction = expanded(self.video)
        for key in bounded_prediction:
            torch.testing.assert_close(
                expanded_prediction[key], bounded_prediction[key], rtol=0, atol=0
            )

    def test_residual_contract_can_exceed_old_appearance_bounds(self) -> None:
        model = FactorizedStructuralJewelEncoder(
            grid_spec=self.spec, slots_per_cell=2, model_dim=8,
            appearance_dim=8, appearance_hidden=16,
            appearance_contract="residual",
        )
        with torch.no_grad():
            model.appearance_residual_head.bias[:3].fill_(2.0)
            model.appearance_residual_head.bias[3:].fill_(0.75)
        prediction = model(self.video)
        self.assertTrue(bool((prediction["colors"] > 1.0).all()))
        self.assertTrue(bool((prediction["color_grads"] > 0.25).all()))
        self.assertTrue(bool((prediction["appearance_residual"][:, :3] == 2.0).all()))
        self.assertEqual(model.model_args["appearance_contract"], "residual")

    def test_native_neighborhood_observes_spatial_and_temporal_context(self) -> None:
        video = torch.zeros(3, 5, 5, 3)
        video[..., 0] = torch.arange(5)[None, None]
        video[..., 1] = torch.arange(5)[None, :, None]
        video[..., 2] = torch.arange(3)[:, None, None]
        samples = sample_native_neighborhood(
            video,
            torch.zeros(1, 3),
            spatial_radius_pixels=1,
            temporal_radius_frames=1,
        )
        self.assertEqual(samples.shape, (1, 7, 3))
        self.assertLess(float(samples[0, 1, 0]), float(samples[0, 2, 0]))
        self.assertLess(float(samples[0, 3, 1]), float(samples[0, 4, 1]))
        self.assertLess(float(samples[0, 5, 2]), float(samples[0, 6, 2]))

    def test_local_adapter_expansion_preserves_residual_checkpoint_exactly(self) -> None:
        torch.manual_seed(17)
        residual = FactorizedStructuralJewelEncoder(
            grid_spec=self.spec, slots_per_cell=2, model_dim=8,
            appearance_dim=8, appearance_hidden=16,
            appearance_contract="residual",
        ).eval()
        expanded = FactorizedStructuralJewelEncoder(
            grid_spec=self.spec, slots_per_cell=2, model_dim=8,
            appearance_dim=8, appearance_hidden=16,
            appearance_contract="local_adapter", appearance_adapter_hidden=24,
        ).eval()
        expanded.load_local_adapter_expansion(residual.state_dict())
        residual_prediction = residual(self.video)
        expanded_prediction = expanded(self.video)
        for key in residual_prediction:
            torch.testing.assert_close(
                expanded_prediction[key], residual_prediction[key], rtol=0, atol=0
            )
        self.assertEqual(
            float(
                expanded_prediction["appearance_adapter_residual"]
                .detach().abs().sum()
            ),
            0.0,
        )

    def test_derivative_features_are_zero_without_a_neighborhood(self) -> None:
        center = torch.rand(4, 1, 3)
        collapsed = center.expand(-1, 7, -1)
        derivative = native_neighborhood_derivatives(collapsed)
        self.assertEqual(derivative.shape, (4, 12))
        torch.testing.assert_close(derivative, torch.zeros_like(derivative))

    def test_derivative_adapter_is_forced_to_use_local_evidence(self) -> None:
        residual = FactorizedStructuralJewelEncoder(
            grid_spec=self.spec, slots_per_cell=2, model_dim=8,
            appearance_dim=8, appearance_hidden=16,
            appearance_contract="residual",
        ).eval()
        derivative = FactorizedStructuralJewelEncoder(
            grid_spec=self.spec, slots_per_cell=2, model_dim=8,
            appearance_dim=8, appearance_hidden=16,
            appearance_contract="derivative_adapter",
            appearance_adapter_radius=0,
            appearance_adapter_temporal_radius=0,
            appearance_adapter_derivative_scale=32,
        ).eval()
        derivative.load_local_adapter_expansion(residual.state_dict())
        with torch.no_grad():
            derivative.appearance_local_adapter[-1].weight.fill_(1.0)
        prediction = derivative(self.video)
        self.assertEqual(
            float(
                prediction["appearance_adapter_residual"].detach().abs().sum()
            ),
            0.0,
        )
        self.assertEqual(
            derivative.model_args["appearance_adapter_derivative_scale"], 32.0
        )

    def test_adapter_only_freeze_has_exact_parameter_ownership(self) -> None:
        model = FactorizedStructuralJewelEncoder(
            grid_spec=self.spec, slots_per_cell=2, model_dim=8,
            appearance_dim=8, appearance_hidden=16,
            appearance_contract="local_adapter",
        )
        model.freeze_base_for_local_adapter()
        trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(all(
            name.startswith("appearance_local_adapter.") for name in trainable
        ))


if __name__ == "__main__":
    unittest.main()

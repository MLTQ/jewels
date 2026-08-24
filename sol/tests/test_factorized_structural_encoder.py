"""Tests for factorized irregular geometry and appearance."""

from __future__ import annotations

import unittest

import torch

from sol.factorized_structural_encoder import FactorizedStructuralJewelEncoder
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
        self.assertEqual(model.model_args["appearance_contract"], "residual")


if __name__ == "__main__":
    unittest.main()

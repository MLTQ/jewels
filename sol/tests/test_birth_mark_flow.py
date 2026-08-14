"""Conditional jewel birth-mark flow tests."""

from __future__ import annotations

import math
import unittest

import torch

from sol.birth_mark_flow import (
    BirthMarkFlowModel,
    birth_mark_flow_objective,
    project_birth_topology,
    rasterize_noisy_marks,
    sample_birth_marks,
)
from sol.splat_density import temporal_standard_deviation
from sol.token_grid import GridSpec


class BirthMarkFlowTests(unittest.TestCase):
    def test_noisy_raster_retains_cell_statistics(self) -> None:
        values = torch.arange(66, dtype=torch.float32).reshape(3, 22)
        cells = torch.tensor([0, 0, 2])
        raster = rasterize_noisy_marks(values, cells, 4)
        self.assertEqual(raster.shape, (4, 46))
        self.assertTrue(torch.equal(raster[0, :22], values[:2].mean(0)))
        self.assertAlmostEqual(float(raster[0, 44]), math.log(3), places=5)
        self.assertEqual(float(raster[1, 45]), 0.0)

    def test_flow_objective_backpropagates_and_samples(self) -> None:
        spec = GridSpec((2, 2, 2), 8)
        model = BirthMarkFlowModel(
            model_dim=32,
            grid_spec=spec,
            context_depth=1,
            noisy_depth=1,
            cell_depth=1,
            mark_depth=1,
            text_dim=6,
        )
        context = torch.randn(spec.n_cells, 46)
        target = torch.randn(7, 22)
        cells = torch.tensor([0, 0, 1, 2, 3, 5, 7])
        slots = torch.tensor([0, 1, 0, 0, 0, 0, 0])
        text = torch.randn(6)
        loss = birth_mark_flow_objective(
            model,
            context,
            target,
            cells,
            slots,
            text,
            noise=torch.randn_like(target),
            flow_time=torch.tensor([0.4]),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        sample = sample_birth_marks(
            model,
            context,
            cells,
            slots,
            text,
            steps=2,
            generator=torch.Generator().manual_seed(3),
        )
        self.assertEqual(sample.shape, target.shape)

    def test_coupled_model_is_exact_base_at_augmentation(self) -> None:
        torch.manual_seed(7)
        spec = GridSpec((2, 2, 2), 8)
        base = BirthMarkFlowModel(
            model_dim=32,
            grid_spec=spec,
            context_depth=0,
            noisy_depth=0,
            cell_depth=1,
            mark_depth=1,
            text_dim=6,
        )
        torch.nn.init.normal_(base.velocity_head.weight, std=0.02)
        coupled = BirthMarkFlowModel(
            model_dim=32,
            grid_spec=spec,
            context_depth=0,
            noisy_depth=0,
            cell_depth=1,
            mark_depth=1,
            text_dim=6,
            set_depth=1,
            set_raster_depth=0,
        )
        incompatible = coupled.load_state_dict(base.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all(name.startswith("set_blocks.") for name in incompatible.missing_keys)
        )
        context = torch.randn(spec.n_cells, 46)
        values = torch.randn(5, 22)
        cells = torch.tensor([0, 0, 1, 4, 7])
        slots = torch.tensor([0, 1, 0, 0, 0])
        arguments = (
            context,
            values,
            torch.tensor([0.3]),
            cells,
            slots,
            torch.randn(6),
        )
        self.assertTrue(torch.equal(base(*arguments), coupled(*arguments)))

    def test_optional_video_guide_preserves_flow_shape(self) -> None:
        spec = GridSpec((2, 2, 2), 8)
        model = BirthMarkFlowModel(
            model_dim=32,
            grid_spec=spec,
            context_depth=1,
            noisy_depth=1,
            guide_depth=1,
            cell_depth=1,
            mark_depth=1,
            text_dim=6,
            guide_dim=3,
        )
        values = torch.randn(4, 22)
        cells = torch.tensor([0, 2, 4, 6])
        velocity = model(
            torch.randn(spec.n_cells, 46),
            values,
            torch.tensor([0.5]),
            cells,
            torch.zeros(4, dtype=torch.long),
            torch.randn(6),
            guide_raster=torch.randn(spec.n_cells, 3),
        )
        self.assertEqual(velocity.shape, values.shape)

    def test_multiscale_guide_attention_backpropagates(self) -> None:
        spec = GridSpec((2, 2, 2), 8)
        model = BirthMarkFlowModel(
            model_dim=32,
            grid_spec=spec,
            context_depth=1,
            noisy_depth=1,
            cell_depth=1,
            mark_depth=1,
            text_dim=6,
            guide_token_dim=16,
            guide_heads=4,
        )
        target = torch.randn(4, 22)
        cells = torch.tensor([0, 2, 4, 6])
        tokens = torch.randn(spec.n_cells, 3, 16)
        loss = birth_mark_flow_objective(
            model,
            torch.randn(spec.n_cells, 46),
            target,
            cells,
            torch.zeros(4, dtype=torch.long),
            torch.randn(6),
            noise=torch.randn_like(target),
            flow_time=torch.tensor([0.5]),
            guide_tokens=tokens,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.guide_attention.parameters()
            )
        )

    def test_hybrid_raster_and_token_guides_share_one_flow(self) -> None:
        spec = GridSpec((2, 2, 2), 8)
        model = BirthMarkFlowModel(
            model_dim=32,
            grid_spec=spec,
            context_depth=1,
            noisy_depth=1,
            guide_depth=1,
            cell_depth=1,
            mark_depth=1,
            text_dim=6,
            guide_dim=3,
            guide_token_dim=16,
            guide_heads=4,
        )
        values = torch.randn(4, 22)
        cells = torch.tensor([0, 2, 4, 6])
        velocity = model(
            torch.randn(spec.n_cells, 46),
            values,
            torch.tensor([0.5]),
            cells,
            torch.zeros(4, dtype=torch.long),
            torch.randn(6),
            guide_raster=torch.randn(spec.n_cells, 3),
            guide_tokens=torch.randn(spec.n_cells, 3, 16),
        )
        self.assertEqual(velocity.shape, values.shape)

    def test_projection_enforces_spatial_and_support_start_cells(self) -> None:
        spec = GridSpec((2, 2, 2), 8)
        features = torch.zeros(2, 22)
        features[:, :3] = torch.tensor([[0.9, 0.9, -0.8], [-0.9, -0.9, 1.8]])
        features[:, 3] = 2 * math.log(0.05)
        features[:, 6] = 2 * math.log(0.05)
        features[:, 8] = 2 * math.log(0.04)
        cells = torch.tensor([0, 7])
        projected = project_birth_topology(
            features,
            cells,
            spec=spec,
            support_sigma=3.0,
            stride_frames=8,
        )
        self.assertTrue((projected[0, :2] < 0).all())
        self.assertTrue((projected[1, :2] > 0).all())
        start = projected[:, 2] - 3 * temporal_standard_deviation(projected)
        first_active = (start * 8).ceil().long()
        time_cells = (first_active * 2 // 8).clamp(0, 1)
        self.assertTrue(torch.equal(time_cells, cells % 2))

    def test_projection_preserves_valid_exterior_edge_centers(self) -> None:
        spec = GridSpec((2, 2, 2), 8)
        features = torch.zeros(2, 22)
        features[:, :2] = torch.tensor([[-1.2, -1.3], [1.2, 1.3]])
        features[:, 3] = 2 * math.log(0.05)
        features[:, 6] = 2 * math.log(0.05)
        features[:, 8] = 2 * math.log(0.04)
        features[:, 2] = torch.tensor([-0.05, 0.6]) + 3 * 0.04
        projected = project_birth_topology(
            features,
            torch.tensor([0, 7]),
            spec=spec,
            support_sigma=3.0,
            stride_frames=8,
        )
        self.assertLess(float((projected - features).abs().max()), 1e-5)

    def test_projection_can_preserve_censored_initial_support(self) -> None:
        spec = GridSpec((2, 2, 2), 8)
        features = torch.zeros(2, 22)
        features[:, :2] = -0.5
        features[:, 3] = 2 * math.log(0.05)
        features[:, 6] = 2 * math.log(0.05)
        features[:, 8] = 2 * math.log(0.04)
        features[:, 2] = -0.5 + 3 * 0.04
        cells = torch.tensor([0, 1])
        strict = project_birth_topology(
            features,
            cells,
            spec=spec,
            support_sigma=3.0,
            stride_frames=8,
        )
        censored = project_birth_topology(
            features,
            cells,
            spec=spec,
            support_sigma=3.0,
            stride_frames=8,
            allow_prefrontier_support=True,
        )
        strict_start = strict[:, 2] - 3 * temporal_standard_deviation(strict)
        censored_start = censored[:, 2] - 3 * temporal_standard_deviation(censored)
        self.assertGreater(float(strict_start[0]), -0.126)
        self.assertAlmostEqual(float(censored_start[0]), -0.5, places=5)
        self.assertAlmostEqual(
            float(censored_start[1]), float(strict_start[1]), places=5
        )

    def test_projection_backpropagates_without_inplace_versions(self) -> None:
        spec = GridSpec((2, 2, 2), 8)
        features = torch.zeros(2, 22)
        features[:, :3] = torch.tensor([[0.9, 0.9, -0.8], [-0.9, -0.9, 1.8]])
        features[:, 3] = 2 * math.log(0.05)
        features[:, 6] = 2 * math.log(0.05)
        features[:, 8] = 2 * math.log(0.04)
        features.requires_grad_()
        projected = project_birth_topology(
            features,
            torch.tensor([0, 7]),
            spec=spec,
            support_sigma=3.0,
            stride_frames=8,
        )
        projected.square().mean().backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())


if __name__ == "__main__":
    unittest.main()

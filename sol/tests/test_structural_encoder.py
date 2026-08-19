"""Tests for the structural (Path B) jewel encoder."""

from __future__ import annotations

import unittest

import torch

from sol.render import render_exact
from sol.structural_encoder import (
    StructuralJewelEncoder,
    precision_factor,
    quaternion_to_matrix,
    render_structural,
)
from sol.token_grid import GridSpec


def _model(slots: int = 5) -> StructuralJewelEncoder:
    torch.manual_seed(0)
    return StructuralJewelEncoder(
        grid_spec=GridSpec((4, 4, 2), 1024), slots_per_cell=slots, model_dim=32
    )


class StructuralEncoderTests(unittest.TestCase):
    def test_quaternion_to_matrix_is_orthonormal(self) -> None:
        q = torch.randn(16, 4)
        r = quaternion_to_matrix(q)
        identity = torch.eye(3).expand(16, 3, 3)
        self.assertTrue(torch.allclose(r @ r.transpose(1, 2), identity, atol=1e-5))

    def test_precision_factor_expresses_extreme_anisotropy(self) -> None:
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        log_scale = torch.tensor([[0.0, -3.0, -3.0]])  # 20x elongation
        factor = precision_factor(q, log_scale)
        precision = factor @ factor.transpose(1, 2)
        covariance = torch.linalg.inv(precision)
        eigenvalues = torch.linalg.eigvalsh(covariance)[0]
        anisotropy = (eigenvalues[-1] / eigenvalues[0]).sqrt()
        self.assertGreater(float(anisotropy), 15.0)

    def test_render_matches_canonical_renderer(self) -> None:
        model = _model()
        video = torch.rand(6, 24, 32, 3)
        prediction = model(video)
        points = torch.rand(48, 3) * 2 - 1
        ours = render_structural(prediction, points, point_chunk=16)
        features = model.canonical_features(prediction)
        reference = render_exact(
            features, points, background=prediction["background"]
        )
        self.assertTrue(torch.allclose(ours, reference, atol=3e-3))

    def test_budget_is_scarce_and_shapes_are_right(self) -> None:
        model = _model(slots=5)
        prediction = model(torch.rand(6, 24, 32, 3))
        self.assertEqual(model.n_jewels, 4 * 4 * 2 * 5)
        self.assertEqual(prediction["centers"].shape, (model.n_jewels, 3))
        self.assertEqual(prediction["precision_factor"].shape, (model.n_jewels, 3, 3))

    def test_no_video_colour_lookup(self) -> None:
        """Colours must come from the network, not from sampling the video."""
        model = _model()
        base = torch.zeros(6, 24, 32, 3)
        shifted = base.clone()
        shifted[..., 0] = 1.0  # drastically change video colour content
        with torch.no_grad():
            first = model(base)["colors"]
            second = model(shifted)["colors"]
        # colours may respond through the trunk, but must not equal the video
        # sample: a lookup would make the red channel saturate to the input.
        self.assertLess(float(second[..., 0].mean()), 0.95)

    def test_gradients_reach_shape_parameters(self) -> None:
        model = _model()
        prediction = model(torch.rand(6, 24, 32, 3))
        points = torch.rand(32, 3) * 2 - 1
        rendered = render_structural(prediction, points, point_chunk=16)
        (rendered - torch.rand_like(rendered)).square().mean().backward()
        grad = model.head.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()

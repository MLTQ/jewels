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
    stratified_slot_offsets,
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

    def test_support_render_matches_five_sigma_oracle_and_gradients(self) -> None:
        torch.manual_seed(14)
        model = _model(slots=2)
        prediction = model(torch.rand(6, 24, 32, 3))
        first = {
            key: value.detach().clone().requires_grad_()
            for key, value in prediction.items()
        }
        sparse = {
            key: value.detach().clone().requires_grad_()
            for key, value in first.items()
        }
        points = torch.rand(51, 3) * 2 - 1
        delta = points[:, None] - first["centers"][None]
        projected = torch.einsum(
            "nij,mnj->mni", first["precision_factor"].transpose(1, 2), delta
        )
        mahalanobis = projected.square().sum(-1)
        logits = (
            -0.5 * mahalanobis
            + torch.nn.functional.logsigmoid(first["logit_w"])[None]
        ).masked_fill(mahalanobis > 25.0, -torch.inf)
        colour = first["colors"][None] + torch.einsum(
            "nij,mnj->mni", first["color_grads"], delta
        )
        oracle = (
            logits.exp()[..., None] * colour
        ).sum(1) + first["background"]
        actual = render_structural(
            sparse,
            points,
            point_chunk=17,
            cull_mode="support_tiled",
            support_capacity=model.n_jewels,
        )
        torch.testing.assert_close(actual, oracle, atol=2e-6, rtol=2e-6)
        oracle.square().sum().backward()
        actual.square().sum().backward()
        for key in first:
            torch.testing.assert_close(
                sparse[key].grad,
                first[key].grad,
                atol=2e-5,
                rtol=2e-5,
                msg=key,
            )

    def test_budget_is_scarce_and_shapes_are_right(self) -> None:
        model = _model(slots=5)
        prediction = model(torch.rand(6, 24, 32, 3))
        self.assertEqual(model.n_jewels, 4 * 4 * 2 * 5)
        self.assertEqual(prediction["centers"].shape, (model.n_jewels, 3))
        self.assertEqual(prediction["precision_factor"].shape, (model.n_jewels, 3, 3))

    def test_non_cubic_slot_offsets_stay_inside_cell(self) -> None:
        offsets = stratified_slot_offsets(36)
        self.assertEqual(offsets.shape, (36, 3))
        self.assertTrue(bool((offsets.abs() < 0.5).all()))
        self.assertEqual(len(torch.unique(offsets, dim=0)), 36)

    def test_video_seeded_colour_tracks_continuous_centres(self) -> None:
        model = StructuralJewelEncoder(
            grid_spec=GridSpec((4, 4, 2), 1024),
            slots_per_cell=5,
            model_dim=32,
            seed_video_colors=True,
        )
        video = torch.zeros(6, 24, 32, 3)
        video[..., :16, 2] = 0.9
        video[..., 16:, 0] = 0.9
        prediction = model(video)
        left = prediction["centers"][:, 0] < 0
        self.assertGreater(float(prediction["colors"][left, 2].mean().detach()), 0.6)
        self.assertGreater(float(prediction["colors"][~left, 0].mean().detach()), 0.6)

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

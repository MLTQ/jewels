"""Tests for the amortized video-to-jewel encoder."""

from __future__ import annotations

import unittest

import torch

from sol.amortized_encoder import (
    VideoToJewelEncoder,
    cholesky_render,
    cholesky_to_log_covariance,
)
from sol.render import render_exact
from sol.token_grid import GridSpec


class AmortizedEncoderTests(unittest.TestCase):
    def test_cholesky_render_matches_canonical_renderer(self) -> None:
        torch.manual_seed(0)
        count = 24
        centers = torch.rand(count, 3) * 2 - 1
        raw = torch.randn(count, 3, 3) * 0.3
        cholesky = torch.tril(raw)
        cholesky[:, range(3), range(3)] = (
            raw[:, range(3), range(3)].abs() + 2.0
        )
        colors = torch.rand(count, 3)
        color_grads = torch.randn(count, 3, 3) * 0.1
        logit_w = torch.randn(count)
        background = torch.rand(3)
        points = torch.rand(64, 3) * 2 - 1
        ours = cholesky_render(
            centers, cholesky, colors, color_grads, logit_w, points, background
        )
        features = torch.cat(
            (
                centers,
                cholesky_to_log_covariance(cholesky),
                colors,
                color_grads.reshape(-1, 9),
                logit_w[:, None],
            ),
            dim=1,
        )
        reference = render_exact(features, points, background=background)
        self.assertTrue(torch.allclose(ours, reference, atol=2e-3))

    def test_encoder_initial_render_tracks_video_structure(self) -> None:
        torch.manual_seed(1)
        model = VideoToJewelEncoder(
            grid_spec=GridSpec((4, 4, 2), 1024), slots_per_cell=8, model_dim=32
        )
        video = torch.zeros(6, 24, 32, 3)
        video[:, :, 16:, 0] = 0.9
        video[:, :, :16, 2] = 0.9
        prediction = model(video)
        points = torch.tensor([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
        rendered = cholesky_render(
            prediction["centers"],
            prediction["cholesky"],
            prediction["colors"],
            prediction["color_grads"],
            prediction["logit_w"],
            points,
            prediction["background"],
        )
        self.assertGreater(float(rendered[0, 0] - rendered[0, 2]), 0.05)
        self.assertGreater(float(rendered[1, 2] - rendered[1, 0]), 0.05)

    def test_canonical_features_shape_and_gradients_flow(self) -> None:
        model = VideoToJewelEncoder(
            grid_spec=GridSpec((4, 4, 2), 1024), slots_per_cell=3, model_dim=32
        )
        video = torch.rand(6, 24, 32, 3)
        prediction = model(video)
        points = torch.rand(40, 3) * 2 - 1
        rendered = cholesky_render(
            prediction["centers"],
            prediction["cholesky"],
            prediction["colors"],
            prediction["color_grads"],
            prediction["logit_w"],
            points,
            prediction["background"],
        )
        rendered.square().mean().backward()
        gradients = [
            parameter.grad.abs().sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertGreater(float(sum(gradients)), 0.0)
        features = model.canonical_features(
            {key: value.detach() for key, value in prediction.items()}
        )
        self.assertEqual(features.shape, (4 * 4 * 2 * 3, 22))



class EncodeDecodeSplitTests(unittest.TestCase):
    def _model(self) -> VideoToJewelEncoder:
        torch.manual_seed(4)
        return VideoToJewelEncoder(
            grid_spec=GridSpec((4, 4, 2), 1024), slots_per_cell=8, model_dim=32
        )

    def test_decode_of_encode_equals_forward_exactly(self) -> None:
        model = self._model()
        video = torch.rand(6, 24, 32, 3)
        direct = model(video)
        split = model.decode(model.encode(video))
        for key in direct:
            self.assertTrue(torch.equal(direct[key], split[key]), key)

    def test_latent_shapes_are_generator_ready(self) -> None:
        model = self._model()
        latent = model.encode(torch.rand(6, 24, 32, 3))
        self.assertEqual(latent["cells"].shape, (32, 32))
        self.assertEqual(latent["seed"].shape, (32, 8, 3))

    def test_decode_needs_no_video_and_rejects_bad_latents(self) -> None:
        model = self._model()
        latent = model.encode(torch.rand(6, 24, 32, 3))
        synthetic = {
            "cells": torch.randn_like(latent["cells"]),
            "seed": torch.rand_like(latent["seed"]),
        }
        prediction = model.decode(synthetic)
        self.assertEqual(prediction["centers"].shape[0], 32 * 8)
        with self.assertRaisesRegex(ValueError, "seed colors"):
            model.decode({"cells": latent["cells"], "seed": latent["seed"][:, :2]})

if __name__ == "__main__":
    unittest.main()

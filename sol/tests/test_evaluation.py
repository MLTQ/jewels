"""Sampled-render validation tests for the autoencoder research gate."""

from __future__ import annotations

import unittest

import torch

from sol.autoencoder import StructuredJewelAutoencoder
from sol.corpus import FeatureNormalizer, FittedExample
from sol.evaluation import evaluate_roundtrip, select_balanced_examples
from sol.synthetic import random_jewels
from sol.token_grid import GridSpec
from sol.train_autoencoder import _sampled_render_loss


class EvaluationTests(unittest.TestCase):
    def test_sampled_render_loss_backpropagates(self) -> None:
        example = FittedExample(
            name="train.pt",
            source_id="train",
            features=random_jewels(12, seed=17),
            background=torch.zeros(3),
            shape=(4, 8, 8),
        )
        normalizer = FeatureNormalizer.fit([example])
        normalized = normalizer.normalize(example.features)[None]
        model = StructuredJewelAutoencoder(
            feature_dim=22,
            model_dim=16,
            latent_dim=8,
            spec=GridSpec((2, 2, 1), slots_per_cell=8),
            enc_depth=1,
            dec_depth=1,
            heads=4,
        )
        target = model.grid.pack_compact(normalized)
        output = model(normalized)
        loss = _sampled_render_loss(
            output.features, target, normalizer, points_per_example=8
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_selection_round_robins_across_sources(self) -> None:
        examples = []
        for source in ("a", "b"):
            for window in range(3):
                examples.append(
                    FittedExample(
                        name=f"{source}_{window}.pt",
                        source_id=source,
                        features=random_jewels(2, seed=window),
                        background=torch.zeros(3),
                        shape=(4, 8, 8),
                    )
                )
        selected = select_balanced_examples(examples, 3)
        self.assertEqual(
            [(example.source_id, example.name) for example in selected],
            [("a", "a_0.pt"), ("b", "b_0.pt"), ("a", "a_1.pt")],
        )

    def test_untrained_model_produces_finite_report(self) -> None:
        example = FittedExample(
            name="held_out.pt",
            source_id="held_out",
            features=random_jewels(12, seed=13),
            background=torch.zeros(3),
            shape=(4, 8, 8),
        )
        normalizer = FeatureNormalizer.fit([example])
        model = StructuredJewelAutoencoder(
            feature_dim=22,
            model_dim=16,
            latent_dim=8,
            spec=GridSpec((2, 2, 1), slots_per_cell=8),
            enc_depth=1,
            dec_depth=1,
            heads=4,
        )
        report = evaluate_roundtrip(
            model,
            [example],
            normalizer,
            device="cpu",
            points_per_example=16,
            max_examples=1,
        )
        self.assertTrue(torch.isfinite(torch.tensor(report.mean_psnr)))
        self.assertEqual(len(report.examples), 1)
        self.assertEqual(report.examples[0].source_id, "held_out")
        self.assertEqual(report.macro_source_psnr, report.mean_psnr)
        self.assertEqual(set(report.source_mean_psnr), {"held_out"})


if __name__ == "__main__":
    unittest.main()

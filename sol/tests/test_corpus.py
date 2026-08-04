"""Leakage and normalization tests for fitted-corpus adaptation."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from sol.corpus import (
    FeatureNormalizer,
    FittedExample,
    load_fitted_corpus,
    split_by_source,
)
from sol.synthetic import random_jewels


def _example(source: str, seed: int) -> FittedExample:
    return FittedExample(
        name=f"{source}_{seed}.pt",
        source_id=source,
        features=random_jewels(12, seed=seed),
        background=torch.zeros(3),
        shape=(4, 8, 8),
    )


def _domain_example(source: str, seed: int, domain_id: str) -> FittedExample:
    example = _example(source, seed)
    example.domain_id = domain_id
    return example


def _write_checkpoint(
    root: Path,
    name: str,
    *,
    shape: tuple[int, int, int],
    source: str,
) -> None:
    jewels = 8
    state = {
        "mu": torch.zeros(jewels, 3),
        "log_scale": torch.zeros(jewels, 3),
        "quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(jewels, 1),
        "color": torch.zeros(jewels, 3),
        "color_grad": torch.zeros(jewels, 3, 3),
        "logit_w": torch.zeros(jewels),
    }
    torch.save(
        {
            "state": state,
            "info": {"background": [0.0, 0.0, 0.0], "shape": shape},
            "source": {"video": source},
        },
        root / name,
    )


class CorpusTests(unittest.TestCase):
    def test_source_split_has_no_video_leakage(self) -> None:
        examples = [
            _example(source, seed)
            for seed, source in enumerate(["a", "a", "b", "b", "c", "c"])
        ]
        split = split_by_source(examples, validation_sources=1, seed=2)
        train_sources = {example.source_id for example in split.train}
        validation_sources = {example.source_id for example in split.validation}
        self.assertTrue(train_sources.isdisjoint(validation_sources))

    def test_normalization_roundtrip_preserves_physical_centers(self) -> None:
        examples = [_example("a", 1), _example("b", 2)]
        normalizer = FeatureNormalizer.fit(examples)
        features = examples[0].features
        normalized = normalizer.normalize(features)
        torch.testing.assert_close(normalized[:, :3], features[:, :3])
        torch.testing.assert_close(normalizer.denormalize(normalized), features)

    def test_explicit_holdout_leaves_other_domains_in_training(self) -> None:
        examples = [
            _domain_example("03", 1, "avenue"),
            _domain_example("05", 2, "avenue"),
            _domain_example("01", 3, "avenue"),
            _domain_example("basketball", 4, "ucf"),
        ]
        split = split_by_source(examples, held_out_sources=["03", "05"])
        self.assertEqual(split.validation_sources, ("03", "05"))
        self.assertEqual({example.source_id for example in split.validation}, {"03", "05"})
        self.assertEqual({example.source_id for example in split.train}, {"01", "basketball"})

    def test_domain_balanced_normalizer_equal_weights_domains(self) -> None:
        numerous = [
            FittedExample(
                name=f"a_{index}",
                source_id="a",
                features=torch.full((2, 22), 2.0),
                background=torch.zeros(3),
                shape=(4, 8, 8),
                domain_id="avenue",
            )
            for index in range(10)
        ]
        sparse = FittedExample(
            name="u_0",
            source_id="u",
            features=torch.full((2, 22), 10.0),
            background=torch.zeros(3),
            shape=(4, 8, 8),
            domain_id="ucf",
        )
        normalizer = FeatureNormalizer.fit(numerous + [sparse], balance_domains=True)
        torch.testing.assert_close(normalizer.mean[3:], torch.full((19,), 6.0))
        torch.testing.assert_close(normalizer.std[3:], torch.full((19,), 4.0))
        torch.testing.assert_close(normalizer.mean[:3], torch.zeros(3))
        torch.testing.assert_close(normalizer.std[:3], torch.ones(3))

    def test_loader_accepts_mixed_shapes_across_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            avenue = root / "avenue"
            ucf = root / "ucf"
            avenue.mkdir()
            ucf.mkdir()
            _write_checkpoint(
                avenue,
                "01_w000000.pt",
                shape=(64, 160, 284),
                source="01.avi",
            )
            _write_checkpoint(
                ucf,
                "basketball_w000000.pt",
                shape=(64, 160, 213),
                source="basketball.avi",
            )

            examples = load_fitted_corpus([avenue, ucf])

        self.assertEqual([example.domain_id for example in examples], ["avenue", "ucf"])
        self.assertEqual([example.shape for example in examples], [(64, 160, 284), (64, 160, 213)])


if __name__ == "__main__":
    unittest.main()

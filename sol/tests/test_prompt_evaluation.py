"""Tests for unseen prompt-template geometry evaluation."""

from __future__ import annotations

from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from sol.prompt_embeddings import build_prompt_cache, collect_prompts
from sol.prompt_evaluation import evaluate_prompt_geometry
from sol.ucf_prompt_manifest import VideoCandidate, build_manifest


class PromptEvaluationTests(unittest.TestCase):
    def test_held_out_templates_retrieve_training_class_centroids(self) -> None:
        classes = ["Basketball", "HorseRiding"]
        selected = [
            VideoCandidate(
                Path(f"/{name}/v_{name}_g{group:02d}_c01.avi"),
                name,
                group,
                1,
                120,
            )
            for name in classes
            for group in range(1, 5)
        ]
        manifest = build_manifest(selected, classes)
        prompts = collect_prompts(manifest)
        embeddings = []
        for prompt in prompts:
            if "basketball" in prompt:
                value = torch.tensor([1.0, 0.05, 0.0])
            else:
                value = torch.tensor([0.05, 1.0, 0.0])
            embeddings.append(F.normalize(value, dim=0))
        cache = build_prompt_cache(manifest, prompts, torch.stack(embeddings))
        report = evaluate_prompt_geometry(manifest, cache)
        self.assertEqual(report.accuracy, 1.0)
        self.assertGreater(report.minimum_margin, 0.8)
        self.assertEqual(
            {metric.predicted_class for metric in report.classes}, set(classes)
        )

    def test_rejects_embedding_cache_from_different_manifest(self) -> None:
        selected = [
            VideoCandidate(
                Path(f"/Basketball/v_Basketball_g{group:02d}_c01.avi"),
                "Basketball",
                group,
                1,
                120,
            )
            for group in range(1, 5)
        ]
        manifest = build_manifest(selected, ["Basketball"])
        prompts = collect_prompts(manifest)
        cache = build_prompt_cache(manifest, prompts, torch.eye(len(prompts)))
        changed = dict(manifest)
        changed["frames"] = 64
        with self.assertRaises(ValueError):
            evaluate_prompt_geometry(changed, cache)


if __name__ == "__main__":
    unittest.main()

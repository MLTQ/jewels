"""Frozen-tokenizer cache restoration tests."""

from __future__ import annotations

from pathlib import Path
import unittest

import torch

from sol.autoencoder import StructuredJewelAutoencoder
from sol.cache_latents import _expand_prompt_conditioned_latents, _restore_tokenizer
from sol.corpus import FittedExample
from sol.prompt_embeddings import build_prompt_cache, collect_prompts
from sol.sparse_autoencoder import SparseJewelAutoencoder
from sol.token_grid import GridSpec
from sol.ucf_prompt_manifest import VideoCandidate, build_manifest


class CacheLatentsTests(unittest.TestCase):
    def _checkpoint(self, sparse: bool) -> tuple[dict, GridSpec]:
        spec = GridSpec((2, 2, 1), 8)
        model_args = {
            "feature_dim": 22,
            "model_dim": 16,
            "latent_dim": 8,
            "enc_depth": 0,
            "dec_depth": 1,
            "heads": 4,
        }
        model = (
            SparseJewelAutoencoder(
                **model_args, spec=spec, encoder_mode="rank"
            )
            if sparse
            else StructuredJewelAutoencoder(**model_args, spec=spec)
        )
        if sparse:
            model_args["encoder_mode"] = "rank"
        return {
            "model": model.state_dict(),
            "meta": {
                "architecture": (
                    "sparse_variable_count_v1" if sparse else "structured_slots_v1"
                ),
                "model_args": model_args,
            },
        }, spec

    def test_restores_sparse_checkpoint_architecture(self) -> None:
        checkpoint, spec = self._checkpoint(sparse=True)
        self.assertIsInstance(
            _restore_tokenizer(checkpoint, spec), SparseJewelAutoencoder
        )

    def test_restores_structured_checkpoint_architecture(self) -> None:
        checkpoint, spec = self._checkpoint(sparse=False)
        self.assertIsInstance(
            _restore_tokenizer(checkpoint, spec), StructuredJewelAutoencoder
        )

    def test_prompt_expansion_uses_train_templates_and_heldout_template(self) -> None:
        candidates = [
            VideoCandidate(
                Path(f"/Basketball/v_Basketball_g{group:02d}_c01.avi"),
                "Basketball",
                group,
                1,
                120,
            )
            for group in range(1, 5)
        ]
        manifest = build_manifest(candidates, ["Basketball"])
        prompts = collect_prompts(manifest)
        prompt_cache = build_prompt_cache(manifest, prompts, torch.eye(4))
        examples = [
            FittedExample(
                name=f"v_Basketball_g{group:02d}_c01_w000000.pt",
                source_id=f"v_Basketball_g{group:02d}_c01",
                features=torch.zeros(2, 22),
                background=torch.zeros(3),
                shape=(4, 4, 4),
            )
            for group in range(1, 5)
        ]
        latents = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
        train_mask = torch.tensor([True, True, True, False])
        expanded = _expand_prompt_conditioned_latents(
            examples, latents, train_mask, manifest, prompt_cache
        )
        expanded_latents, conditions, names, sources, expanded_train = expanded
        self.assertEqual(expanded_latents.shape, (10, 2, 3))
        self.assertEqual(conditions.shape, (10, 4))
        self.assertEqual(int(expanded_train.sum()), 9)
        torch.testing.assert_close(expanded_latents[:3], latents[0].expand(3, -1, -1))
        torch.testing.assert_close(conditions[:3], torch.eye(4)[:3])
        torch.testing.assert_close(conditions[-1], torch.eye(4)[3])
        self.assertTrue(names[-1].endswith("prompt_index=3"))
        self.assertEqual(sources[-1], "v_Basketball_g04_c01")

        with self.assertRaisesRegex(ValueError, "split disagrees"):
            _expand_prompt_conditioned_latents(
                examples,
                latents,
                torch.tensor([True, True, False, False]),
                manifest,
                prompt_cache,
            )


if __name__ == "__main__":
    unittest.main()

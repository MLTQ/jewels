"""Tests for topology trainer split utilities and count priors."""

from __future__ import annotations

import unittest

import torch

from sol.scaffold_topology_data import ScaffoldTopologyView
from sol.streaming_corpus import PromptedField
from sol.streaming_data import BirthTarget
from sol.train_scaffold_topology import (
    PreparedTopologySource,
    PreparedTopologyView,
    _control_views,
    _diagnostic_split_override,
    _flatten,
    _mean_counts_by_index,
)


def _source(split: str, class_id: int, counts: torch.Tensor) -> PreparedTopologySource:
    births = BirthTarget(
        values=torch.zeros(int(counts.sum()), 22),
        cell_indices=torch.repeat_interleave(torch.arange(len(counts)), counts),
        slot_indices=torch.cat([torch.arange(int(value)) for value in counts]),
        counts=counts,
        global_ids=torch.arange(int(counts.sum())),
        birth_frames=torch.zeros(int(counts.sum()), dtype=torch.long),
    )
    topology = ScaffoldTopologyView(
        index=0,
        frontier=0,
        commit_stop=4,
        carried_global_features=torch.empty(0, 22),
        carried_ids=torch.empty(0, dtype=torch.long),
        births=births,
        birth_global_features=torch.zeros(int(counts.sum()), 22),
        target_active_global_features=torch.zeros(int(counts.sum()), 22),
        active_commit_ids=torch.arange(int(counts.sum())),
    )
    field = PromptedField(
        source_id=f"{split}-{class_id}",
        class_id=class_id,
        class_name=f"class-{class_id}",
        split=split,
        features=torch.zeros(int(counts.sum()), 22),
        frames=4,
        train_prompt_indices=(),
        evaluation_prompt_indices=(),
    )
    return PreparedTopologySource(
        field,
        (
            PreparedTopologyView(
                topology,
                torch.full((len(counts), 3), 0.25 * (class_id + 1)),
                torch.zeros(len(counts), 3),
            ),
        ),
    )


class TrainScaffoldTopologyTests(unittest.TestCase):
    def test_split_flatten_mean_and_control_ownership(self) -> None:
        sources = (
            _source("train", 0, torch.tensor([1, 3, 0, 2])),
            _source("train", 1, torch.tensor([3, 1, 2, 0])),
            _source("validation", 0, torch.tensor([2, 2, 1, 1])),
        )
        train = _flatten(sources, "train")
        validation = _flatten(sources, "validation")
        self.assertEqual(len(train), 2)
        self.assertEqual(len(validation), 1)
        mean = _mean_counts_by_index(train)
        self.assertTrue(torch.equal(mean[0], torch.tensor([2.0, 2.0, 1.0, 1.0])))
        controls = _control_views(validation)
        self.assertEqual(controls[0].source_id, "validation-0")
        self.assertTrue(torch.equal(controls[0].target_counts, torch.tensor([2, 2, 1, 1])))

    def test_diagnostic_split_override_is_source_exact(self) -> None:
        sources = (
            _source("train", 0, torch.tensor([1, 2, 0, 1])),
            _source("validation", 1, torch.tensor([2, 1, 1, 0])),
            _source("validation", 2, torch.tensor([1, 1, 1, 1])),
        )
        held_out = sources[1].field.source_id
        overridden = _diagnostic_split_override(sources, [held_out])
        self.assertEqual(
            [source.field.split for source in overridden],
            ["train", "validation", "train"],
        )
        self.assertEqual(sources[2].field.split, "validation")

    def test_diagnostic_split_override_rejects_invalid_sets(self) -> None:
        sources = (
            _source("train", 0, torch.tensor([1, 2, 0, 1])),
            _source("validation", 1, torch.tensor([2, 1, 1, 0])),
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            _diagnostic_split_override(sources, ["train-0", "train-0"])
        with self.assertRaisesRegex(ValueError, "unknown"):
            _diagnostic_split_override(sources, ["missing"])
        with self.assertRaisesRegex(ValueError, "retain"):
            _diagnostic_split_override(sources, ["train-0", "validation-1"])


if __name__ == "__main__":
    unittest.main()

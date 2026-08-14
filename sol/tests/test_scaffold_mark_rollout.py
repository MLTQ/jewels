"""Tests for autonomous initial-plus-two-continuation mark rollout."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from sol.birth_mark_flow import BirthMarkFlowModel
from sol.scaffold_mark_rollout import rollout_scaffold_marks
from sol.scaffold_topology import ScaffoldTopologyModel
from sol.scaffold_topology_realizer import realize_topology_marks
from sol.streaming_data import FeatureStandardizer
from sol.token_grid import GridSpec


class ScaffoldMarkRolloutTests(unittest.TestCase):
    def test_three_strides_use_append_only_generated_state(self) -> None:
        torch.manual_seed(2)
        spec = GridSpec((2, 2, 2), 4)
        topology = ScaffoldTopologyModel(
            model_dim=8,
            grid_spec=spec,
            encoder_depth=1,
            cell_depth=1,
        )
        with torch.no_grad():
            for parameter in topology.parameters():
                parameter.zero_()
            topology.occupancy_head.bias.fill_(10)
            topology.positive_count_head.bias.fill_(-10)
        flow = BirthMarkFlowModel(
            model_dim=8,
            grid_spec=spec,
            context_depth=1,
            noisy_depth=1,
            guide_depth=1,
            cell_depth=1,
            mark_depth=1,
            text_dim=8,
            guide_dim=3,
            guide_heads=1,
        )
        standardizer = FeatureStandardizer(torch.zeros(22), torch.ones(22))
        guides = [torch.rand(spec.n_cells, 3) for _ in range(3)]
        text = torch.randn(8)
        with patch(
            "sol.scaffold_mark_rollout.realize_topology_marks",
            wraps=realize_topology_marks,
        ) as realize:
            rollout = rollout_scaffold_marks(
                topology,
                flow,
                guides,
                text,
                standardizer,
                standardizer,
                total_frames=24,
                stride_frames=8,
                support_sigma=2.0,
                topology_spec=spec,
                occupancy_threshold=0.5,
                device="cpu",
                steps=1,
                generator=torch.Generator().manual_seed(5),
            )
        self.assertEqual(
            [call.kwargs["allow_prefrontier_support"] for call in realize.call_args_list],
            [True, False, False],
        )
        self.assertEqual(rollout.completed_frames, 24)
        self.assertEqual(len(rollout.windows), 3)
        self.assertEqual([window.born_jewels for window in rollout.windows], [8, 8, 8])
        self.assertTrue(torch.equal(rollout.global_ids, torch.arange(24)))
        self.assertTrue(all(window.max_prior_feature_error == 0 for window in rollout.windows))
        self.assertTrue(all(window.max_carried_feature_error == 0 for window in rollout.windows))

        with patch(
            "sol.scaffold_mark_rollout.realize_topology_marks",
            wraps=realize_topology_marks,
        ) as strict_realize:
            rollout_scaffold_marks(
                topology,
                flow,
                guides[:1],
                text,
                standardizer,
                standardizer,
                total_frames=24,
                stride_frames=8,
                support_sigma=2.0,
                topology_spec=spec,
                occupancy_threshold=0.5,
                device="cpu",
                steps=1,
                generator=torch.Generator().manual_seed(5),
                allow_initial_prefrontier=False,
            )
        self.assertFalse(
            strict_realize.call_args.kwargs["allow_prefrontier_support"]
        )

    def test_external_topology_counts_bypass_feedback_prediction(self) -> None:
        torch.manual_seed(3)
        spec = GridSpec((2, 2, 2), 4)
        topology = ScaffoldTopologyModel(
            model_dim=8,
            grid_spec=spec,
            encoder_depth=0,
            cell_depth=0,
        )
        flow = BirthMarkFlowModel(
            model_dim=8,
            grid_spec=spec,
            context_depth=0,
            noisy_depth=0,
            guide_depth=0,
            cell_depth=0,
            mark_depth=0,
            text_dim=8,
            guide_dim=3,
            guide_heads=1,
        )
        standardizer = FeatureStandardizer(torch.zeros(22), torch.ones(22))
        guides = [torch.rand(spec.n_cells, 3) for _ in range(3)]
        fixed = []
        for cell in (0, 3, 7):
            counts = torch.zeros(spec.n_cells, dtype=torch.long)
            counts[cell] = 1
            fixed.append(counts)
        with patch(
            "sol.scaffold_mark_rollout.predict_realizer_topology",
            side_effect=AssertionError("topology prediction must be bypassed"),
        ):
            rollout = rollout_scaffold_marks(
                topology,
                flow,
                guides,
                torch.randn(8),
                standardizer,
                standardizer,
                total_frames=24,
                stride_frames=8,
                support_sigma=2.0,
                topology_spec=spec,
                occupancy_threshold=0.5,
                device="cpu",
                steps=1,
                generator=torch.Generator().manual_seed(9),
                owned_counts=fixed,
            )
        self.assertEqual(
            rollout.topology_contract, "externally_owned_cell_counts"
        )
        self.assertTrue(
            all(
                torch.equal(actual, expected)
                for actual, expected in zip(rollout.counts, fixed)
            )
        )
        self.assertEqual([window.born_jewels for window in rollout.windows], [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
